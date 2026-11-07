"""Exact Contracts v1 acceptance rule, independent of solver and filesystem."""

from nsu_budget_agent.contracts.models import (
    AcceptanceDecision,
    AttemptResult,
    ContractError,
    ValidationResult,
)


class V1AcceptancePolicy:
    def decide(self, result: AttemptResult, validation: ValidationResult) -> AcceptanceDecision:
        if result.status != "candidate" or result.snapshot_sha256 is None:
            raise ContractError("acceptance requires a candidate snapshot")
        mandatory = {"format", "paths", "protected_files", "snapshot"}
        checks = {check.check_id: check for check in validation.checks}
        accepted = (
            result.attempt_id == validation.attempt_id
            and result.snapshot_sha256 == validation.snapshot_sha256
            and mandatory.issubset(checks)
            and all(checks[name].blocking for name in mandatory)
            and all(
                check.status != "failed" and (not check.blocking or check.status == "passed")
                for check in validation.checks
            )
        )
        return AcceptanceDecision(
            attempt_id=result.attempt_id,
            snapshot_sha256=result.snapshot_sha256,
            status="accept" if accepted else "reject",
            reason="v1 checks passed" if accepted else "v1 checks rejected",
        )
