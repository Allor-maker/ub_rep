"""Trusted provider attestations. Observing a few replies cannot verify bounds."""

import importlib
import re
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

from nsu_budget_agent.contracts.models import ContractError, ModelId, ProviderProfile

from .gateway import InputBound


@dataclass(frozen=True, kw_only=True)
class ProviderCompatibility:
    """Supplied by trusted deployment code, never by tasks or smoke observations.

    evidence identifies the authoritative guarantees and reviewed bound adapter.
    It is a review reference, not proof that the supplied implementation is sound.
    """

    input_bound: InputBound
    profiles: Mapping[ModelId, ProviderProfile]
    evidence: str
    output_token_parameter: Literal["max_tokens", "max_completion_tokens"] = "max_tokens"

    def __post_init__(self) -> None:
        if (
            not callable(self.input_bound)
            or not isinstance(self.evidence, str)
            or not self.evidence.strip()
        ):
            raise ContractError("trusted provider evidence and input bound are required")
        if self.output_token_parameter not in ("max_tokens", "max_completion_tokens"):
            raise ContractError("invalid provider output parameter")
        object.__setattr__(self, "profiles", MappingProxyType(dict(self.profiles)))

    def require_verified(self, models: tuple[ModelId, ...]) -> None:
        for model in models:
            profile = self.profiles.get(model)
            if not isinstance(profile, ProviderProfile) or (
                not profile.input_bound_verified
                or not profile.total_output_cap_verified
                or profile.reasoning_mode == "unverified"
            ):
                raise ContractError("strict operation requires confirmed provider guarantees")


def load_compatibility(reference: str) -> ProviderCompatibility:
    """Explicit module:factory reference from trusted operator configuration only."""
    if not re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*:[A-Za-z_]\w*", reference):
        raise ContractError("a trusted compatibility adapter module:factory is required")
    module, name = reference.split(":")
    try:
        value = getattr(importlib.import_module(module), name)()
    except Exception:
        # Adapter exceptions can contain environment secrets. Never expose them.
        raise ContractError("provider compatibility adapter could not be loaded") from None
    if not isinstance(value, ProviderCompatibility):
        raise ContractError("provider compatibility adapter returned an invalid type")
    return value
