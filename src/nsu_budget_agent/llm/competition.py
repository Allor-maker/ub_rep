"""Explicit competition risk policy. Estimates and reported usage are not attestations."""

import json
from dataclasses import replace

from nsu_budget_agent.contracts.models import (
    ErrorInfo,
    Id,
    LLMRequest,
    ModelId,
    OperationFailed,
    UsageRecord,
)


def wire_body(request: LLMRequest, parameter: str = "max_tokens") -> bytes:
    return json.dumps(
        {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            parameter: request.max_output_tokens,
            "stream": False,
            "n": 1,
        },
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")


def estimate_input(request: LLMRequest, body: bytes) -> int:
    """ASCII-escaped wire bytes + 1024 + 256/message, deliberately an estimate.

    Covers much more than content/4 heuristics; no claim about unknown server
    normalization/framing. No model/tokenizer download or learned estimator.
    """
    return len(body) + 1024 + 256 * len(request.messages)


def fit_request(
    request: LLMRequest,
    available_units: int,
    *,
    min_output: int = 64,
    parameter: str = "max_tokens",
) -> tuple[LLMRequest, bytes, int]:
    """Choose an output cap fitting both A2 budgets before creating a lease.

    Re-serialize after cap changes. There is still exactly one physical request.
    """
    for _ in range(3):
        body = wire_body(request, parameter)
        estimate = estimate_input(request, body)
        cap = min(request.max_output_tokens, max(0, (available_units - estimate) // 10))
        if cap < min_output:
            raise OperationFailed(
                ErrorInfo(
                    code="budget_exhausted",
                    message="competition estimate cannot fit minimum output",
                )
            )
        if cap == request.max_output_tokens:
            return request, body, estimate
        request = replace(request, max_output_tokens=cap)
    raise OperationFailed(
        ErrorInfo(code="budget_exhausted", message="competition estimate does not fit")
    )


def _counter(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def normalize_competition_usage(
    payload: object, *, request_id: Id, attempt_id: Id, model: ModelId, success: bool
) -> UsageRecord:
    """Settle only an unambiguous interpretation of the complete reported counters.

    `exact` is arithmetic settlement of reported usage under this opt-in policy,
    not a verified assertion about undisclosed provider billing. Missing or
    contradictory data stays unknown and retains the estimated lease in A2.
    """
    data = payload.get("usage") if isinstance(payload, dict) else None
    counts = data if isinstance(data, dict) else {}
    prompt, completion = (
        _counter(counts.get("prompt_tokens")),
        _counter(counts.get("completion_tokens")),
    )
    details = counts.get("completion_tokens_details")
    nested = details.get("reasoning_tokens") if isinstance(details, dict) else None
    top = counts.get("reasoning_tokens")
    reasoning = _counter(nested if nested is not None else top)
    total_raw = counts.get("total_tokens")
    total = _counter(total_raw)
    malformed = (
        (details is not None and not isinstance(details, dict))
        or (nested is not None and _counter(nested) is None)
        or (top is not None and _counter(top) is None)
        or (nested is not None and top is not None and nested != top)
        or (total_raw is not None and total is None)
    )
    cache_details = counts.get("prompt_tokens_details")
    if isinstance(cache_details, dict) and "cached_tokens" in cache_details:
        cached = _counter(cache_details["cached_tokens"])
        malformed |= cached is None or prompt is None or cached > prompt
    elif cache_details is not None and not isinstance(cache_details, dict):
        malformed = True
    reasoning_text = False
    choices = payload.get("choices") if isinstance(payload, dict) else None
    if isinstance(choices, list):
        for choice in choices:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict):
                reasoning_text |= bool(message.get("reasoning_content") or message.get("reasoning"))
    exact = False
    output = completion
    if success and not malformed and prompt is not None and completion is not None:
        if reasoning == 0 and not reasoning_text:
            exact = total is None or total == prompt + completion
        elif reasoning is not None and reasoning > 0 and total is not None:
            if total == prompt + completion and reasoning <= completion:
                exact = True  # completion already includes reasoning.
            elif total == prompt + completion + reasoning:
                output = completion + reasoning
                exact = True
        elif reasoning is None and not reasoning_text and model.endswith("-noreason"):
            exact = total is not None and total == prompt + completion
    return UsageRecord(
        request_id=request_id,
        attempt_id=attempt_id,
        model=model,
        status="exact" if exact else "unknown",
        input_tokens=prompt,
        output_tokens=output,
        reasoning_tokens=reasoning,
    )
