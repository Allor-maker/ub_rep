"""Normalize only the explicitly configured usage semantics, never response text."""

from nsu_budget_agent.contracts.models import Id, ModelId, ProviderProfile, UsageRecord


def _counter(value: object) -> int | None:
    return value if type(value) is int and value >= 0 else None


def normalize_usage(
    payload: object,
    *,
    request_id: Id,
    attempt_id: Id,
    model: ModelId,
    profile: ProviderProfile,
    success: bool,
) -> UsageRecord:
    usage = payload.get("usage") if isinstance(payload, dict) else None
    data = usage if isinstance(usage, dict) else {}
    prompt = _counter(data.get("prompt_tokens"))
    completion = _counter(data.get("completion_tokens"))
    details = data.get("completion_tokens_details")
    nested = details.get("reasoning_tokens") if isinstance(details, dict) else None
    top = data.get("reasoning_tokens")
    reasoning = _counter(nested if nested is not None else top)
    malformed = (
        (details is not None and not isinstance(details, dict))
        or (nested is not None and _counter(nested) is None)
        or (top is not None and _counter(top) is None)
        or (nested is not None and top is not None and nested != top)
    )
    exact = success and not malformed and prompt is not None and completion is not None
    output = completion
    if profile.reasoning_mode == "separate":
        exact &= reasoning is not None
        if completion is not None and reasoning is not None:
            output = completion + reasoning
    elif profile.reasoning_mode == "included":
        exact &= reasoning is None or (completion is not None and reasoning <= completion)
    else:
        exact = False
    # Error-envelope counters are retained as diagnostics, not proof of free usage.
    return UsageRecord(
        request_id=request_id,
        attempt_id=attempt_id,
        model=model,
        status="exact" if exact else "unknown",
        input_tokens=prompt,
        output_tokens=output,
        reasoning_tokens=reasoning,
    )
