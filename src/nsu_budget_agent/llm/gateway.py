"""One POST per step, with dispatch accounting and conservative failure handling."""

import asyncio
import json
import math
import os
from collections.abc import Callable, Mapping
from dataclasses import replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from types import TracebackType
from typing import Literal, cast, get_args
from urllib.parse import urlsplit
from uuid import uuid4

import httpx

from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    ContractError,
    ErrorInfo,
    Id,
    LLMReply,
    LLMRequest,
    ModelId,
    OperationFailed,
    ProviderProfile,
    UsageRecord,
    _id,
    _instance,
    _int,
    _number,
)
from nsu_budget_agent.contracts.ports import BudgetController, LLMSession
from nsu_budget_agent.telemetry.llm import LLMEvent

from .usage import normalize_usage

InputBound = Callable[[LLMRequest, bytes], int]
EventSink = Callable[[LLMEvent], None]


def _failure(
    code: str, message: str, *, retryable: bool = False, retry_after_s: float | None = None
) -> OperationFailed:
    return OperationFailed(
        ErrorInfo(code=code, message=message, retryable=retryable, retry_after_s=retry_after_s)
    )


def _retry_after(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            timestamp = parsedate_to_datetime(value)
            if timestamp.tzinfo is None:
                return None
            seconds = (timestamp - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, OverflowError, TypeError):
            return None
    return max(0.0, seconds) if math.isfinite(seconds) else None


def _json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _invalid_constant(value: str) -> None:
    raise ValueError("nonfinite JSON constant")


class _Session:
    def __init__(self, gateway: "NeuralDeepGateway", attempt_id: Id, budget: AttemptBudget) -> None:
        self._gateway = gateway
        self._attempt_id = attempt_id
        self._budget = budget
        self._used = False

    async def complete(self, request: LLMRequest) -> LLMReply:
        self._gateway._check_open()
        if self._used:
            raise ContractError("only one complete call is allowed per attempt step")
        self._used = True  # Before any await, including failures and concurrent calls.
        _instance(request, LLMRequest, "request")
        return await self._gateway._complete(self._attempt_id, self._budget, request)


class NeuralDeepGateway:
    """Construct and close in the same loop as the BudgetController.

    input_bound must include serialized chat framing/provider overhead, per model.
    Profiles and bounds are trusted configuration, never inferred by this gateway.
    The optional HTTP transport is owned/closed by the gateway and must not retry.
    """

    def __init__(
        self,
        *,
        budget: BudgetController,
        input_bound: InputBound,
        profiles: Mapping[ModelId, ProviderProfile] | None = None,
        accounting_mode: Literal["strict", "best_effort"] = "strict",
        max_concurrent_requests: int = 2,
        requests_per_minute: int = 60,
        timeout_s: float = 30.0,
        max_response_bytes: int = 1_048_576,
        output_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens",
        transport: httpx.AsyncBaseTransport | None = None,
        environ: Mapping[str, str] | None = None,
        event_sink: EventSink | None = None,
        competition_mode: bool = False,
    ) -> None:
        _int(max_concurrent_requests, "max_concurrent_requests", 1)
        _int(requests_per_minute, "requests_per_minute", 1)
        _int(max_response_bytes, "max_response_bytes", 1)
        _number(timeout_s, "timeout_s", positive=True)
        if max_concurrent_requests > 5 or requests_per_minute > 60:
            raise ContractError("parallel/RPM limits exceed the configured competition ceiling")
        if accounting_mode not in ("strict", "best_effort"):
            raise ContractError("invalid accounting mode")
        if type(competition_mode) is not bool or (
            competition_mode and accounting_mode != "best_effort"
        ):
            raise ContractError("competition policy requires explicit best_effort accounting")
        if output_token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ContractError("invalid output token parameter")
        if not callable(input_bound):
            raise ContractError("input_bound must be supplied by trusted configuration")
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise ContractError("gateway requires a running event loop") from exc
        environment = os.environ if environ is None else environ
        base = environment.get("LLM_BASE_URL", "").strip()
        key = environment.get("LLM_API_KEY", "")
        try:
            parsed = urlsplit(base)
            valid = (
                parsed.scheme == "https"
                and bool(parsed.hostname)
                and parsed.port != 0
                and not parsed.username
                and not parsed.password
                and not parsed.query
                and not parsed.fragment
                and not any(ord(char) < 32 for char in base)
            )
        except ValueError:
            valid = False
        if not valid or not key.strip() or any(ord(char) < 32 or ord(char) > 126 for char in key):
            raise ContractError("valid HTTPS LLM_BASE_URL and nonempty API key are required")
        self._endpoint = base.rstrip("/") + "/chat/completions"
        self._key = key
        self._budget = budget
        self._input_bound = input_bound
        self._profiles = dict(profiles or {})
        for model, profile in self._profiles.items():
            if model not in get_args(ModelId):
                raise ContractError("unsupported model profile")
            _instance(profile, ProviderProfile, "profile")
        self._strict = accounting_mode == "strict"
        self.competition_mode = competition_mode
        self.competition_overruns: set[Id] = set()
        self.events: list[LLMEvent] = []
        self._event_started: dict[Id, float] = {}
        self._output_parameter = output_token_parameter
        self._timeout_s = timeout_s
        self._response_limit = max_response_bytes
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._send_lock = asyncio.Lock()
        self._interval_s = 60.0 / requests_per_minute
        self._next_send_at = 0.0
        self._cooldown_until = 0.0
        self._sessions: dict[tuple[Id, int], _Session] = {}
        self._attempt_budgets: dict[Id, AttemptBudget] = {}
        self._closed = False
        self._event_sink = event_sink
        # Low-level transport: no client redirect/auth loop, hooks or HTTPX URL logs.
        self._transport = (
            transport
            if transport is not None
            else httpx.AsyncHTTPTransport(
                retries=0,
                trust_env=False,
                limits=httpx.Limits(
                    max_connections=max_concurrent_requests,
                    max_keepalive_connections=max_concurrent_requests,
                ),
            )
        )

    def _check_open(self) -> None:
        try:
            current = asyncio.get_running_loop()
        except RuntimeError as exc:
            raise ContractError("gateway requires its running event loop") from exc
        if current is not self._loop or self._closed:
            raise ContractError("gateway is closed or used from another loop")

    def bind(self, attempt_id: Id, step_index: int, budget: AttemptBudget) -> LLMSession:
        self._check_open()
        _id(attempt_id, "attempt_id")
        _int(step_index, "step_index")
        _instance(budget, AttemptBudget, "budget")
        if step_index >= budget.max_steps:
            raise ContractError("step_index exceeds attempt budget")
        if attempt_id in self._attempt_budgets and self._attempt_budgets[attempt_id] != budget:
            raise ContractError("attempt is already bound to a different budget")
        self._budget.open_attempt(attempt_id, budget)
        self._attempt_budgets[attempt_id] = budget
        key = (attempt_id, step_index)
        if key not in self._sessions:
            self._sessions[key] = _Session(self, attempt_id, budget)
        return self._sessions[key]

    def _emit(self, event: LLMEvent) -> None:
        now = self._loop.time()
        if event.phase == "reserved":
            self._event_started[event.request_id] = now
        event = replace(
            event, elapsed_s=max(0.0, now - self._event_started.get(event.request_id, now))
        )
        self.events.append(event)
        if self._event_sink is not None:
            try:
                self._event_sink(event)
            except Exception:
                pass  # A telemetry sink must not change billing or mask cancellation.

    async def _wait_rate_slot(self) -> None:
        async with self._send_lock:
            while True:
                now = self._loop.time()
                delay = max(self._next_send_at, self._cooldown_until) - now
                if delay <= 0:
                    self._next_send_at = now + self._interval_s
                    return
                await asyncio.sleep(delay)

    async def _read_response(self, response: httpx.Response) -> bytes:
        if response.is_stream_consumed:
            if len(response.content) > self._response_limit:
                raise _failure("invalid_output", "provider response exceeds size limit")
            return response.content
        chunks = bytearray()
        async for chunk in response.aiter_raw():
            if len(chunks) + len(chunk) > self._response_limit:
                raise _failure("invalid_output", "provider response exceeds size limit")
            chunks.extend(chunk)
        return bytes(chunks)

    def _http_error(self, response: httpx.Response, payload: object) -> OperationFailed | None:
        error = payload.get("error") if isinstance(payload, dict) else None
        if 200 <= response.status_code < 300 and error is None:
            return None
        code = error.get("code") if isinstance(error, dict) else None
        if code in ("insufficient_quota", "quota_exhausted", "budget_exhausted") or (
            self.competition_mode
            and (code in ("quota_exceeded", "insufficient_balance") or response.status_code == 402)
        ):
            failure = _failure("quota_exhausted", "provider quota exhausted")
            self._budget.halt(failure.error)
            return failure
        if response.status_code == 429:
            delay = _retry_after(response.headers.get("retry-after"))
            return _failure(
                "rate_limited", "provider rate limit", retryable=True, retry_after_s=delay
            )
        if response.status_code in (401, 403):
            failure = _failure("denied", "provider authorization denied")
            self._budget.halt(failure.error)
            return failure
        return _failure(
            "provider_error",
            "provider rejected the request",
            retryable=response.status_code >= 500 or response.status_code == 408,
        )

    async def _complete(
        self, attempt_id: Id, budget: AttemptBudget, request: LLMRequest
    ) -> LLMReply:
        if request.model not in budget.allowed_models:
            raise ContractError("model is not allowed by attempt budget")
        profile = self._profiles.get(request.model, ProviderProfile())
        if self._strict and (
            not profile.input_bound_verified
            or not profile.total_output_cap_verified
            or profile.reasoning_mode == "unverified"
        ):
            raise _failure("unsupported", "strict accounting requires a verified provider profile")
        body = json.dumps(
            {
                "model": request.model,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in request.messages
                ],
                self._output_parameter: request.max_output_tokens,
                "stream": False,
                "n": 1,
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        bound = self._input_bound(request, body)
        _int(bound, "input_bound_tokens")
        if self.competition_mode:
            from .competition import fit_request

            available = min(
                self._budget.snapshot().available_units,
                self._budget.snapshot(attempt_id).available_units,
            )
            request, body, bound = fit_request(request, available, parameter=self._output_parameter)
        request_id = uuid4().hex
        unknown = UsageRecord(
            request_id=request_id,
            attempt_id=attempt_id,
            model=request.model,
            status="unknown",
            input_tokens=None,
            output_tokens=None,
        )
        event = LLMEvent(
            request_id=request_id,
            attempt_id=attempt_id,
            model=request.model,
            phase="reserved",
            input_bound_tokens=bound,
            output_cap_tokens=request.max_output_tokens,
        )
        lease = self._budget.reserve(
            request_id, attempt_id, request.model, bound, request.max_output_tokens
        )
        dispatched = accounted = False
        try:
            self._emit(event)
            deadline = min(budget.deadline_monotonic, self._loop.time() + self._timeout_s)
            async with asyncio.timeout_at(deadline):
                async with self._semaphore:
                    await self._wait_rate_slot()
                    self._check_open()
                    remaining = max(0.001, deadline - self._loop.time())
                    http_request = httpx.Request(
                        "POST",
                        self._endpoint,
                        content=body,
                        headers={
                            "Authorization": "Bearer " + self._key,
                            "Content-Type": "application/json",
                            "Accept-Encoding": "identity",
                        },
                        extensions={
                            "timeout": {
                                name: remaining for name in ("connect", "read", "write", "pool")
                            }
                        },
                    )
                    self._budget.mark_dispatched(request_id)
                    dispatched = True
                    self._emit(replace(event, phase="dispatched"))
                    response = await self._transport.handle_async_request(http_request)
                    # No await between receiving headers and updating the shared limiter.
                    if response.status_code == 429:
                        delay = _retry_after(response.headers.get("retry-after"))
                        self._cooldown_until = max(
                            self._cooldown_until,
                            self._loop.time() + max(self._interval_s, delay or 0.0),
                        )
                    response_read = False
                    try:
                        try:
                            raw = await self._read_response(response)
                            response_read = True
                        except OperationFailed:
                            # Status/Retry-After remain meaningful even if a body is too large.
                            failed_read = self._http_error(response, None)
                            if failed_read is not None:
                                raise failed_read from None
                            raise
                    finally:
                        if response_read:
                            await response.aclose()
                        # After an interrupted read, arbitrary stream cleanup can
                        # swallow cancellation forever. Do not start it here or
                        # orphan a task; transport ownership remains with the
                        # gateway and its eventual aclose().
                    try:
                        payload = json.loads(
                            raw, object_pairs_hook=_json_object, parse_constant=_invalid_constant
                        )
                    except (ValueError, UnicodeError, RecursionError):
                        payload = None
                    failed = self._http_error(response, payload)
                    if self.competition_mode:
                        from .competition import normalize_competition_usage

                        usage = normalize_competition_usage(
                            payload,
                            request_id=request_id,
                            attempt_id=attempt_id,
                            model=request.model,
                            success=failed is None,
                        )
                    else:
                        usage = normalize_usage(
                            payload,
                            request_id=request_id,
                            attempt_id=attempt_id,
                            model=request.model,
                            profile=profile,
                            success=failed is None,
                        )
                    try:
                        self._budget.record_usage(usage)
                        accounted = True
                    except OperationFailed as exc:
                        # A2 commits the full exact charge and halts before reporting
                        # a reservation breach. The received result may still be
                        # independently validated without another paid request.
                        if not (
                            self.competition_mode
                            and failed is None
                            and usage.status == "exact"
                            and usage.cost_units is not None
                            and usage.cost_units > lease.reserved_units
                            and exc.error.code == "accounting_violation"
                            and self._budget.snapshot().blocked
                        ):
                            raise
                        accounted = True
                        self.competition_overruns.add(attempt_id)
                    if self.competition_mode and usage.status == "unknown":
                        self._budget.halt(
                            ErrorInfo(
                                code="unknown_usage",
                                message="competition reported usage is uncertain; paid routes stopped",
                            )
                        )
                    if (
                        (
                            usage.input_tokens is not None
                            and usage.input_tokens > lease.input_bound_tokens
                        )
                        or (
                            usage.output_tokens is not None
                            and usage.output_tokens > lease.output_cap_tokens
                        )
                        or (
                            usage.reasoning_tokens is not None
                            and usage.reasoning_tokens > lease.output_cap_tokens
                        )
                    ):
                        failure = _failure(
                            "accounting_violation",
                            "reported usage exceeded competition estimate/cap"
                            if self.competition_mode
                            else "provider exceeded a token bound",
                        )
                        self._budget.halt(failure.error)
                        if self.competition_mode and failed is None and usage.status == "exact":
                            self.competition_overruns.add(attempt_id)
                        if not (
                            self.competition_mode and failed is None and usage.status == "exact"
                        ):
                            raise failure
                    if failed is not None:
                        raise failed
                    reply = self._reply(payload, request_id, usage)
                    self._emit(
                        replace(
                            event,
                            phase="replied",
                            usage_status=usage.status,
                            input_tokens=usage.input_tokens,
                            output_tokens=usage.output_tokens,
                        )
                    )
                    return reply
        except asyncio.CancelledError:
            self._emit(
                replace(event, phase="cancelled", usage_status="unknown" if dispatched else None)
            )
            raise
        except (TimeoutError, httpx.TimeoutException):
            self._emit(replace(event, phase="failed", error_code="timeout"))
            raise _failure("timeout", "LLM operation timed out", retryable=True) from None
        except (httpx.InvalidURL, UnicodeError):
            self._emit(replace(event, phase="failed", error_code="invalid_input"))
            raise _failure("invalid_input", "invalid LLM request configuration") from None
        except httpx.TransportError:
            self._emit(replace(event, phase="failed", error_code="provider_error"))
            raise _failure("provider_error", "LLM transport failed", retryable=True) from None
        except OperationFailed as exc:
            self._emit(replace(event, phase="failed", error_code=exc.error.code))
            raise
        finally:
            if not dispatched:
                self._budget.release_unsent(request_id)
            elif not accounted:
                self._budget.record_usage(unknown)
                if self.competition_mode:
                    self._budget.halt(
                        ErrorInfo(
                            code="unknown_usage",
                            message="competition dispatch outcome unknown; paid routes stopped",
                        )
                    )

    @staticmethod
    def _reply(payload: object, request_id: Id, usage: UsageRecord) -> LLMReply:
        if not isinstance(payload, dict):
            raise _failure("invalid_output", "provider response is not a JSON object")
        choices = payload.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
            raise _failure("invalid_output", "provider response must contain one choice")
        choice = choices[0]
        message = choice.get("message")
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("content"), str)
            or message.get("tool_calls")
            or message.get("function_call")
        ):
            raise _failure("invalid_output", "provider response must contain plain text")
        finish = choice.get("finish_reason")
        return LLMReply(
            request_id=request_id,
            text=message["content"],
            usage=usage,
            finish=cast(
                Literal["stop", "length", "other"],
                finish if finish in ("stop", "length") else "other",
            ),
        )

    async def aclose(self) -> None:
        if not self._closed:
            self._check_open()
            self._closed = True
            await self._transport.aclose()

    async def __aenter__(self) -> "NeuralDeepGateway":
        self._check_open()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()
