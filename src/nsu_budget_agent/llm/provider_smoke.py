"""Explicit, small provider observation through the existing A3 Gateway.

No API call on import, --help, or without --live and an operator-supplied key.
Observations never generate a verified ProviderProfile or a tokenizer bound.
"""

import argparse
import asyncio
import json
import os
from collections.abc import AsyncIterator, Mapping
from typing import Literal, get_args

import httpx

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    ChatMessage,
    ContractError,
    LLMRequest,
    ModelId,
    OperationFailed,
    ProviderProfile,
    ToolLimits,
)

from .compatibility import ProviderCompatibility, load_compatibility
from .gateway import NeuralDeepGateway


def _observe(raw: bytes, output: dict) -> None:
    try:
        payload = json.loads(raw)
    except (ValueError, UnicodeError, RecursionError):
        return
    if not isinstance(payload, dict):
        return
    usage = payload.get("usage")
    output["usage_object_present"] = isinstance(usage, dict)
    if isinstance(usage, dict):
        for key in ("prompt_tokens", "completion_tokens", "reasoning_tokens", "total_tokens"):
            value = usage.get(key)
            output[key] = value if type(value) is int and value >= 0 else None
        details = usage.get("completion_tokens_details")
        nested = details.get("reasoning_tokens") if isinstance(details, dict) else None
        output["nested_reasoning_tokens"] = nested if type(nested) is int and nested >= 0 else None
        output["usage_shape_valid"] = (
            type(usage.get("prompt_tokens")) is int
            and usage["prompt_tokens"] >= 0
            and type(usage.get("completion_tokens")) is int
            and usage["completion_tokens"] >= 0
            and (details is None or isinstance(details, dict))
            and (nested is None or type(nested) is int and nested >= 0)
            and (
                usage.get("reasoning_tokens") is None
                or type(usage["reasoning_tokens"]) is int
                and usage["reasoning_tokens"] >= 0
            )
            and (
                nested is None
                or usage.get("reasoning_tokens") is None
                or nested == usage["reasoning_tokens"]
            )
        )
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        message = choices[0].get("message")
        output["reasoning_text_present"] = isinstance(message, dict) and any(
            bool(message.get(field)) for field in ("reasoning_content", "reasoning")
        )


class _ObservedStream(httpx.AsyncByteStream):
    def __init__(self, original: httpx.AsyncByteStream, output: dict) -> None:
        self.original, self.output = original, output

    async def __aiter__(self) -> AsyncIterator[bytes]:
        raw = bytearray()
        overflow = False
        async for chunk in self.original:
            if len(raw) + len(chunk) <= 65_536:
                raw.extend(chunk)
            else:
                overflow = True
            yield chunk
        if not overflow:
            _observe(bytes(raw), self.output)

    async def aclose(self) -> None:
        await self.original.aclose()


class _ObservingTransport(httpx.AsyncBaseTransport):
    def __init__(self, original: httpx.AsyncBaseTransport) -> None:
        self.original = original
        self.observations: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.original.handle_async_request(request)
        output: dict = {"http_status": response.status_code}
        self.observations.append(output)
        if response.is_stream_consumed:
            if len(response.content) <= 65_536:
                _observe(response.content, output)
        else:
            if not isinstance(response.stream, httpx.AsyncByteStream):
                raise httpx.TransportError("provider transport returned a non-async stream")
            response.stream = _ObservedStream(response.stream, output)
        return response

    async def aclose(self) -> None:
        await self.original.aclose()


async def run_smoke(
    *,
    live: bool,
    budget_units: int,
    environ: Mapping[str, str],
    compatibility: ProviderCompatibility | None = None,
    best_effort_diagnostic: bool = False,
    input_bound_tokens: int | None = None,
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
    models: tuple[ModelId, ...] = get_args(ModelId),
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict:
    if not live or not environ.get("LLM_API_KEY", "").strip():
        raise ContractError("live smoke requires --live and LLM_API_KEY")
    if type(budget_units) is not int or budget_units <= 0:
        raise ContractError("an explicit positive diagnostic budget in internal Units is required")
    if (
        not models
        or len(models) > 4
        or len(set(models)) != len(models)
        or any(model not in get_args(ModelId) for model in models)
    ):
        raise ContractError("select one to four distinct permitted models")
    if best_effort_diagnostic:
        if (
            compatibility is not None
            or type(input_bound_tokens) is not int
            or input_bound_tokens <= 0
        ):
            raise ContractError(
                "best-effort diagnostics require an explicit input estimate, without an attested adapter"
            )
        estimate = input_bound_tokens
        compatibility = ProviderCompatibility(
            input_bound=lambda request, body: estimate,
            profiles={model: ProviderProfile() for model in models},
            evidence="Unverified diagnostic estimate only; not a production attestation",
            output_token_parameter=output_token_parameter,
        )
    else:
        if compatibility is None:
            raise ContractError("strict smoke requires a confirmed compatibility adapter")
        compatibility.require_verified(models)
    deadline = asyncio.get_running_loop().time() + 180
    ledger = BudgetLedger(limit_units=budget_units, deadline_monotonic=deadline)
    budget = AttemptBudget(
        max_cost_units=budget_units,
        max_llm_calls=2 * len(models),
        max_steps=2 * len(models),
        deadline_monotonic=deadline,
        allowed_models=models,
        tool_limits=ToolLimits(
            timeout_s=1,
            max_read_bytes=1,
            max_output_bytes=65_536,
            max_write_bytes=1,
            max_process_memory_bytes=1,
            max_process_count=1,
        ),
    )
    observed = _ObservingTransport(
        transport if transport is not None else httpx.AsyncHTTPTransport(retries=0, trust_env=False)
    )
    rows: list[dict] = []
    async with NeuralDeepGateway(
        budget=ledger,
        input_bound=compatibility.input_bound,
        profiles=compatibility.profiles,
        output_token_parameter=compatibility.output_token_parameter,
        accounting_mode="best_effort" if best_effort_diagnostic else "strict",
        max_concurrent_requests=1,
        timeout_s=20,
        max_response_bytes=65_536,
        transport=observed,
        environ=environ,
    ) as gateway:
        try:
            for model in models:
                for cap, prompt in (
                    (8, "Reply with the digit 2 only."),
                    (64, "List the integers from 1 to 200, in order, separated by spaces."),
                ):
                    row: dict = {"model": model, "output_cap_tokens": cap}
                    count = len(observed.observations)
                    try:
                        reply = await gateway.bind("provider-smoke", len(rows), budget).complete(
                            LLMRequest(
                                model=model,
                                messages=(ChatMessage(role="user", content=prompt),),
                                max_output_tokens=cap,
                            )
                        )
                        row.update(
                            finish=reply.finish,
                            content_nonempty=bool(reply.text),
                            short_answer_matches=reply.text.strip() == "2" if cap == 8 else None,
                            usage_status=reply.usage.status,
                            input_tokens=reply.usage.input_tokens,
                            output_tokens=reply.usage.output_tokens,
                            reasoning_tokens=reply.usage.reasoning_tokens,
                            output_cap_observed_within=(
                                reply.usage.output_tokens <= cap
                                if reply.usage.output_tokens is not None
                                and reply.usage.status == "exact"
                                else None
                            ),
                        )
                    except OperationFailed as exc:
                        row["error_code"] = exc.error.code
                    if len(observed.observations) > count:
                        row["observed"] = observed.observations[-1]
                    if best_effort_diagnostic and "observed" in row:
                        counters = row["observed"]
                        completion = counters.get("completion_tokens")
                        reasoning = counters.get("nested_reasoning_tokens")
                        if reasoning is None:
                            reasoning = counters.get("reasoning_tokens")
                        row["cap_consistent_if_included"] = (
                            completion <= cap if completion is not None else None
                        )
                        row["cap_consistent_if_separate"] = (
                            completion + reasoning <= cap
                            if completion is not None and reasoning is not None
                            else None
                        )
                    rows.append(row)
                    # Diagnostic unknown usage is retained in full by A2. Only
                    # well-formed successful samples can advance to the next
                    # preplanned observation, never retry this request.
                    if not _can_continue(row, best_effort_diagnostic):
                        break
                if not _can_continue(rows[-1], best_effort_diagnostic):
                    break
        finally:
            ledger.close_attempt("provider-smoke")
    snapshot = ledger.snapshot()
    return {
        "strict": not best_effort_diagnostic,
        "profile_promoted": False,
        "observations": rows,
        "spent_units": snapshot.spent_units,
        "held_units": snapshot.held_units,
        "blocked": snapshot.blocked,
        "completed_samples": len(rows) == 2 * len(models)
        and all(_can_continue(row, best_effort_diagnostic) for row in rows),
    }


def _can_continue(row: dict, diagnostic: bool) -> bool:
    if "error_code" in row or row.get("finish") not in ("stop", "length"):
        return False
    if not diagnostic:
        return row.get("usage_status") == "exact"
    return (
        row.get("observed", {}).get("usage_shape_valid") is True
        and row.get("finish") in ("stop", "length")
        and row.get("cap_consistent_if_included") is not False
        and row.get("cap_consistent_if_separate") is not False
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--live", action="store_true", help="explicitly permit up to eight billable calls"
    )
    parser.add_argument(
        "--budget-units", type=int, help="explicit test budget in internal Units, not BUDGET tokens"
    )
    parser.add_argument("--adapter", default=os.environ.get("LLM_COMPATIBILITY_ADAPTER", ""))
    parser.add_argument(
        "--best-effort-diagnostic",
        action="store_true",
        help="research only: no guarantee of the real provider spend; never enables production fallback",
    )
    parser.add_argument(
        "--input-bound-tokens",
        type=int,
        help="explicit unverified prompt estimate for diagnostic mode only",
    )
    parser.add_argument(
        "--output-token-parameter",
        choices=("max_tokens", "max_completion_tokens"),
        default="max_tokens",
    )
    parser.add_argument("--model", choices=get_args(ModelId), action="append")
    args = parser.parse_args(argv)
    try:
        if not args.live or not os.environ.get("LLM_API_KEY", "").strip():
            raise ContractError("live diagnostics disabled")
        if args.budget_units is None:
            raise ContractError("diagnostic budget required")
        if args.best_effort_diagnostic and args.adapter:
            raise ContractError("do not combine a diagnostic estimate and an attested adapter")
        compatibility = None if args.best_effort_diagnostic else load_compatibility(args.adapter)
        report = asyncio.run(
            run_smoke(
                live=args.live,
                budget_units=args.budget_units,
                compatibility=compatibility,
                best_effort_diagnostic=args.best_effort_diagnostic,
                input_bound_tokens=args.input_bound_tokens,
                output_token_parameter=args.output_token_parameter,
                environ=os.environ,
                models=tuple(args.model) if args.model else get_args(ModelId),
            )
        )
    except (ContractError, OperationFailed):
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "reason": "explicit live flag, key, budget and confirmed adapter or separately enabled diagnostic estimate required",
                }
            )
        )
        return 2
    print(json.dumps(report, ensure_ascii=True))
    return 0 if report["completed_samples"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
