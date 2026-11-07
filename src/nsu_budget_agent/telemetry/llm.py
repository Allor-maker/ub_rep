"""Safe LLM lifecycle events. Caller-owned sinks must not attach raw HTTP data."""

from dataclasses import dataclass
from typing import Literal

from nsu_budget_agent.contracts.models import Id, ModelId


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMEvent:
    request_id: Id
    attempt_id: Id
    model: ModelId
    phase: Literal["reserved", "dispatched", "replied", "failed", "cancelled"]
    usage_status: Literal["exact", "unknown"] | None = None
    error_code: str | None = None
