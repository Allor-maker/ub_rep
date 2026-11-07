"""Contracts v1 data and boundary validation. No I/O, SDK or service implementations.

Context-dependent checks (filesystem containment, deadlines at dispatch, registry
uniqueness and state transitions) belong to the implementing ports.
"""

from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass, fields
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Literal, TypeAlias, cast, get_args

Id: TypeAlias = str
Units: TypeAlias = int
RelPath: TypeAlias = PurePosixPath
ModelId: TypeAlias = Literal[
    "qwen3.8-27b",
    "qwen3.8-27b-noreason",
    "qwen3.6-35b-a3b",
    "qwen3.6-35b-a3b-noreason",
]
JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
JsonObject: TypeAlias = dict[str, JsonValue]


class ContractError(ValueError):
    """Invalid DTO or programmer violation of a port invariant."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ContractError(message)


def _string(value: str, name: str, *, nonempty: bool = False) -> None:
    _require(isinstance(value, str), f"{name} must be a string")
    if nonempty:
        _require(bool(value.strip()), f"{name} must not be empty")


def _id(value: str, name: str = "id") -> None:
    _string(value, name, nonempty=True)
    _require(len(value) <= 128, f"{name} exceeds 128 characters")
    _require(
        not any(unicodedata.category(c) == "Cc" for c in value),
        f"{name} contains control characters",
    )


def _int(value: int, name: str, minimum: int = 0) -> None:
    _require(type(value) is int and value >= minimum, f"{name} must be int >= {minimum}")


def _number(
    value: float, name: str, *, minimum: float | None = None, positive: bool = False
) -> None:
    _require(type(value) in (int, float), f"{name} must be a finite number")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    _require(finite, f"{name} must be finite")
    if minimum is not None:
        _require(value >= minimum, f"{name} must be >= {minimum}")
    if positive:
        _require(value > 0, f"{name} must be positive")


def _bool(value: object, name: str) -> None:
    _require(type(value) is bool, f"{name} must be bool")


def _enum(value: object, choices: tuple[str, ...], name: str) -> None:
    _require(isinstance(value, str) and value in choices, f"invalid {name}")


def _instance(value: object, cls: type, name: str) -> None:
    _require(isinstance(value, cls), f"{name} must be {cls.__name__}")


def _items(value: tuple, cls: type, name: str) -> None:
    _require(type(value) is tuple, f"{name} must be a tuple")
    for item in value:
        _instance(item, cls, name)


def _unique(values: tuple, name: str) -> None:
    _require(len(set(values)) == len(values), f"duplicate {name}")


def _ids(values: tuple[str, ...], name: str, *, unique: bool = True) -> None:
    _items(values, str, name)
    for value in values:
        _id(value, name)
    if unique:
        _unique(values, name)


def _sha(value: object, name: str = "sha256") -> None:
    _require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        f"{name} must be 64 lowercase hex characters",
    )


def parse_rel_path(raw: str, *, allow_directory: bool = False) -> RelPath:
    """Validate raw input BEFORE PurePosixPath can normalize unsafe segments.

    Only tool directory arguments may opt into the single '.' path. DTO file
    paths never opt in. Callers doing I/O must still enforce roots and symlinks.
    """
    _string(raw, "relative path", nonempty=True)
    if raw == "." and allow_directory:
        return PurePosixPath(raw)
    _require("\\" not in raw and "\x00" not in raw, "invalid relative path characters")
    _require(
        not PureWindowsPath(raw).drive and not raw.startswith("/"),
        "relative path must not have a root or drive",
    )
    _require(
        all(part not in ("", ".", "..") for part in raw.split("/")),
        "relative path has empty, dot or parent segments",
    )
    return PurePosixPath(raw)


def _rel(value: RelPath) -> None:
    _instance(value, PurePosixPath, "relative path")
    parse_rel_path(value.as_posix())


def _paths(value: tuple[RelPath, ...], name: str, *, unique: bool = True) -> None:
    _items(value, PurePosixPath, name)
    for path in value:
        _rel(path)
    if unique:
        _unique(value, name)


def _absolute(value: Path, name: str) -> None:
    _instance(value, Path, name)
    _require(value.is_absolute(), f"{name} must be absolute")


def _artifacts(value: tuple[ArtifactRef, ...], name: str = "artifacts") -> None:
    _items(value, ArtifactRef, name)
    _unique(tuple(item.path for item in value), name)


class _JsonSnapshot(str):
    """Private immutable storage; public fields always expose ordinary JSON copies."""

    __slots__ = ()


def _restore_json_dto(cls, values):
    # Re-enter validation when unpickling instead of installing mutable slot values.
    return cls(**values)


class _JsonDTO:
    """Keep v1 dict/list field types without retaining mutable containers in a DTO.

    Each JSON field read produces a detached view. Mutating that view (including
    through built-in dict/list methods) cannot change this or another DTO.
    """

    __slots__ = ()

    def __getattribute__(self, name):
        value = object.__getattribute__(self, name)
        return json.loads(value) if type(value) is _JsonSnapshot else value

    def __copy__(self):
        return self

    def __deepcopy__(self, memo):
        return self

    def __reduce_ex__(self, protocol):
        return _restore_json_dto, (
            type(self),
            {f.name: getattr(self, f.name) for f in fields(self)},
        )


def _copy_json(value: object, active: set[int] | None = None) -> JsonValue:
    if value is None or type(value) in (bool, int, str):
        return cast(JsonValue, value)
    if type(value) is float:
        _number(cast(float, value), "JSON number")
        return cast(float, value)
    _require(type(value) in (dict, list), "invalid JSON value")
    active = set() if active is None else active
    _require(id(value) not in active, "cyclic JSON value")
    active.add(id(value))
    try:
        if isinstance(value, dict):
            result_dict: JsonObject = {}
            for key, item in value.items():
                _string(key, "JSON key")
                result_dict[key] = _copy_json(item, active)
            return result_dict
        else:
            return [_copy_json(item, active) for item in cast(list, value)]
    except RecursionError as exc:
        raise ContractError("JSON nesting exceeds interpreter limit") from exc
    finally:
        active.remove(id(value))


def _json_field(dto: object, name: str, *, object_only: bool = False) -> None:
    value = getattr(dto, name)
    if object_only:
        _require(isinstance(value, dict), f"{name} must be a JSON object")
    copied = _copy_json(value)
    try:
        snapshot = json.dumps(copied, allow_nan=False, separators=(",", ":"))
    except (ValueError, RecursionError) as exc:
        raise ContractError("JSON cannot be serialized") from exc
    object.__setattr__(dto, name, _JsonSnapshot(snapshot))


def _schema_refs(value: JsonValue) -> None:
    """Inspect schema positions in JSON Schema drafts 4 through 2020-12.

    This is a reference policy check, not a complete schema validator. Maps of
    property/definition names and annotation/constant values are not schemas.
    Unknown extension keywords are left to the consumer's chosen dialect.
    """
    if not isinstance(value, dict):  # Boolean schemas contain no references.
        return
    for keyword in ("$ref", "$dynamicRef", "$recursiveRef"):
        if keyword in value:
            ref = value[keyword]
            _require(
                isinstance(ref, str) and (not ref or ref.startswith("#")),
                "only local schema $ref values are allowed",
            )
    for keyword in ("properties", "patternProperties", "$defs", "definitions", "dependentSchemas"):
        schemas = value.get(keyword)
        if isinstance(schemas, dict):
            for schema in schemas.values():
                _schema_refs(schema)
    dependencies = value.get("dependencies")
    if isinstance(dependencies, dict):
        for schema in dependencies.values():
            if isinstance(schema, dict):  # Arrays here are property dependencies.
                _schema_refs(schema)
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        schemas = value.get(keyword)
        if isinstance(schemas, list):
            for schema in schemas:
                _schema_refs(schema)
    for keyword in (
        "additionalProperties",
        "additionalItems",
        "contains",
        "propertyNames",
        "not",
        "if",
        "then",
        "else",
        "unevaluatedItems",
        "unevaluatedProperties",
        "contentSchema",
        "items",
    ):
        schema = value.get(keyword)
        if keyword == "items" and isinstance(schema, list):  # Tuple form in older drafts.
            for item in schema:
                _schema_refs(item)
        else:
            _schema_refs(schema)


@dataclass(frozen=True, slots=True, kw_only=True)
class ErrorInfo:
    code: str
    message: str
    retryable: bool = False
    retry_after_s: float | None = None

    def __post_init__(self):
        _string(self.code, "code", nonempty=True)
        _string(self.message, "message")
        _require(len(self.message) <= 2048, "error message exceeds 2048 characters")
        _bool(self.retryable, "retryable")
        if self.retry_after_s is not None:
            _number(self.retry_after_s, "retry_after_s", minimum=0)


class OperationFailed(Exception):
    """Expected port failure, retaining structured error information."""

    def __init__(self, error: ErrorInfo):
        _instance(error, ErrorInfo, "error")
        self.error = error
        super().__init__(error.message)


@dataclass(frozen=True, slots=True, kw_only=True)
class ArtifactRef:
    path: RelPath
    size_bytes: int
    sha256: str
    media_type: str | None = None
    protected: bool = False

    def __post_init__(self):
        _rel(self.path)
        _int(self.size_bytes, "size_bytes")
        _sha(self.sha256)
        if self.media_type is not None:
            _string(self.media_type, "media_type")
        _bool(self.protected, "protected")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskSpec:
    task_id: Id
    instruction: str
    reward: int
    task_root: Path
    work_root: Path
    artifacts: tuple[ArtifactRef, ...]
    type_hint: str | None = None
    check_hint: str | None = None

    def __post_init__(self):
        _id(self.task_id, "task_id")
        _string(self.instruction, "instruction", nonempty=True)
        _int(self.reward, "reward")
        _absolute(self.task_root, "task_root")
        _absolute(self.work_root, "work_root")
        _artifacts(self.artifacts)
        for name in ("type_hint", "check_hint"):
            if getattr(self, name) is not None:
                _string(getattr(self, name), name)


@dataclass(frozen=True, slots=True, kw_only=True)
class SolverOffer:
    task_id: Id
    solver_id: Id
    applicability: Literal["yes", "no", "unknown"]
    required_tools: tuple[str, ...]
    expected_cost_units: Units | None = None
    success_probability: float | None = None
    expected_duration_s: float | None = None
    reason: str = ""

    def __post_init__(self):
        _id(self.task_id, "task_id")
        _id(self.solver_id, "solver_id")
        _enum(self.applicability, ("yes", "no", "unknown"), "applicability")
        _items(self.required_tools, str, "required_tools")
        _unique(self.required_tools, "required_tools")
        if self.expected_cost_units is not None:
            _int(self.expected_cost_units, "expected_cost_units")
        if self.success_probability is not None:
            _number(self.success_probability, "success_probability", minimum=0)
            _require(self.success_probability <= 1, "success_probability exceeds 1")
        if self.expected_duration_s is not None:
            _number(self.expected_duration_s, "expected_duration_s", minimum=0)
        _string(self.reason, "reason", nonempty=self.applicability == "no")


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolLimits:
    timeout_s: float
    max_read_bytes: int
    max_output_bytes: int
    max_write_bytes: int
    max_process_memory_bytes: int
    max_process_count: int

    def __post_init__(self):
        _number(self.timeout_s, "timeout_s", positive=True)
        for name in (
            "max_read_bytes",
            "max_output_bytes",
            "max_write_bytes",
            "max_process_memory_bytes",
            "max_process_count",
        ):
            _int(getattr(self, name), name, 1)


def _attempt_limits(value: AttemptBudget | AttemptLimits) -> None:
    _int(value.max_cost_units, "max_cost_units")
    _int(value.max_llm_calls, "max_llm_calls")
    _int(value.max_steps, "max_steps", 1)
    _items(value.allowed_models, str, "allowed_models")
    _unique(value.allowed_models, "allowed_models")
    for model in value.allowed_models:
        _enum(model, get_args(ModelId), "model")
    _require(
        value.max_llm_calls == 0 or bool(value.allowed_models), "LLM calls require allowed_models"
    )
    _instance(value.tool_limits, ToolLimits, "tool_limits")


@dataclass(frozen=True, slots=True, kw_only=True)
class AttemptBudget:
    max_cost_units: Units
    max_llm_calls: int
    max_steps: int
    deadline_monotonic: float
    allowed_models: tuple[ModelId, ...]
    tool_limits: ToolLimits

    def __post_init__(self):
        _attempt_limits(self)
        _number(self.deadline_monotonic, "deadline_monotonic")


@dataclass(frozen=True, slots=True, kw_only=True)
class UsageRecord:
    request_id: Id
    attempt_id: Id
    model: ModelId
    status: Literal["exact", "unknown"]
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None = None

    def __post_init__(self):
        _id(self.request_id, "request_id")
        _id(self.attempt_id, "attempt_id")
        _enum(self.model, get_args(ModelId), "model")
        _enum(self.status, ("exact", "unknown"), "usage status")
        for name in ("input_tokens", "output_tokens", "reasoning_tokens"):
            if getattr(self, name) is not None:
                _int(getattr(self, name), name)
        if self.status == "exact":
            _require(
                self.input_tokens is not None and self.output_tokens is not None,
                "exact usage requires input and output counters",
            )
            if self.reasoning_tokens is not None:
                _require(
                    self.reasoning_tokens <= self.output_tokens,
                    "reasoning_tokens exceeds full output",
                )

    @property
    def cost_units(self) -> Units | None:
        if self.status == "unknown":
            return None
        return cast(int, self.input_tokens) + 10 * cast(int, self.output_tokens)


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetLease:
    request_id: Id
    attempt_id: Id
    model: ModelId
    input_bound_tokens: int
    output_cap_tokens: int
    state: Literal["reserved", "dispatched", "uncertain", "settled", "released"]
    usage: UsageRecord | None = None

    def __post_init__(self):
        _id(self.request_id, "request_id")
        _id(self.attempt_id, "attempt_id")
        _enum(self.model, get_args(ModelId), "model")
        _int(self.input_bound_tokens, "input_bound_tokens")
        _int(self.output_cap_tokens, "output_cap_tokens", 1)
        _enum(
            self.state,
            ("reserved", "dispatched", "uncertain", "settled", "released"),
            "lease state",
        )
        if self.state in ("settled", "uncertain"):
            _instance(self.usage, UsageRecord, "usage")
            _require(
                self.usage.status == ("exact" if self.state == "settled" else "unknown"),
                "usage status does not match lease state",
            )
            _require(
                (self.usage.request_id, self.usage.attempt_id, self.usage.model)
                == (self.request_id, self.attempt_id, self.model),
                "usage/lease mismatch",
            )
        else:
            _require(self.usage is None, "unsettled/released lease cannot contain usage")

    @property
    def reserved_units(self) -> Units:
        return self.input_bound_tokens + 10 * self.output_cap_tokens


@dataclass(frozen=True, slots=True, kw_only=True)
class BudgetSnapshot:
    limit_units: Units
    spent_units: Units
    held_units: Units
    calls_used: int
    blocked: bool

    def __post_init__(self):
        for name in ("limit_units", "spent_units", "held_units", "calls_used"):
            _int(getattr(self, name), name)
        _bool(self.blocked, "blocked")

    @property
    def available_units(self) -> Units:
        return max(0, self.limit_units - self.spent_units - self.held_units)


@dataclass(frozen=True, slots=True, kw_only=True)
class ProviderProfile:
    reasoning_mode: Literal["included", "separate", "unverified"] = "unverified"
    input_bound_verified: bool = False
    total_output_cap_verified: bool = False

    def __post_init__(self):
        _enum(self.reasoning_mode, ("included", "separate", "unverified"), "reasoning_mode")
        _bool(self.input_bound_verified, "input_bound_verified")
        _bool(self.total_output_cap_verified, "total_output_cap_verified")


@dataclass(frozen=True, slots=True, kw_only=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str

    def __post_init__(self):
        _enum(self.role, ("system", "user", "assistant"), "role")
        _string(self.content, "content")


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMRequest:
    model: ModelId
    messages: tuple[ChatMessage, ...]
    max_output_tokens: int

    def __post_init__(self):
        _enum(self.model, get_args(ModelId), "model")
        _items(self.messages, ChatMessage, "messages")
        _require(bool(self.messages), "messages must not be empty")
        _int(self.max_output_tokens, "max_output_tokens", 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class LLMReply:
    request_id: Id
    text: str
    finish: Literal["stop", "length", "other"]
    usage: UsageRecord

    def __post_init__(self):
        _id(self.request_id, "request_id")
        _string(self.text, "text")
        _enum(self.finish, ("stop", "length", "other"), "finish")
        _instance(self.usage, UsageRecord, "usage")
        _require(self.request_id == self.usage.request_id, "reply/usage request_id mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolSpec(_JsonDTO):
    name: str
    description: str
    input_schema: JsonObject
    output_schema: JsonObject
    mutates_files: bool
    available: bool = True
    unavailable_reason: str | None = None

    def __post_init__(self):
        _string(self.name, "name")
        _string(self.description, "description")
        for name in ("input_schema", "output_schema"):
            _json_field(self, name, object_only=True)
            _schema_refs(getattr(self, name))
        _bool(self.mutates_files, "mutates_files")
        _bool(self.available, "available")
        if self.unavailable_reason is not None or not self.available:
            _string(self.unavailable_reason, "unavailable_reason", nonempty=not self.available)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolCall(_JsonDTO):
    tool_name: str
    arguments: JsonObject

    def __post_init__(self):
        _string(self.tool_name, "tool_name")
        _json_field(self, "arguments", object_only=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolContext:
    attempt_id: Id
    workspace_root: Path
    source_artifacts: tuple[ArtifactRef, ...]
    protected_paths: tuple[RelPath, ...]
    deadline_monotonic: float
    limits: ToolLimits

    def __post_init__(self):
        _id(self.attempt_id, "attempt_id")
        _absolute(self.workspace_root, "workspace_root")
        _artifacts(self.source_artifacts, "source_artifacts")
        _paths(self.protected_paths, "protected_paths")
        _number(self.deadline_monotonic, "deadline_monotonic")
        _instance(self.limits, ToolLimits, "limits")


@dataclass(frozen=True, slots=True, kw_only=True)
class ToolResult(_JsonDTO):
    call_id: Id
    tool_name: str
    status: Literal["ok", "error"]
    data: JsonValue = None
    artifacts: tuple[ArtifactRef, ...] = ()
    protected_paths: tuple[RelPath, ...] = ()
    error: ErrorInfo | None = None
    duration_s: float = 0.0
    truncated: bool = False

    def __post_init__(self):
        _id(self.call_id, "call_id")
        _string(self.tool_name, "tool_name")
        _enum(self.status, ("ok", "error"), "tool status")
        _json_field(self, "data")
        _artifacts(self.artifacts)
        _paths(self.protected_paths, "protected_paths")
        _number(self.duration_s, "duration_s", minimum=0)
        _bool(self.truncated, "truncated")
        if self.status == "ok":
            _require(self.error is None, "ok result cannot contain error")
        else:
            _instance(self.error, ErrorInfo, "error")
            _require(self.data is None, "error result cannot contain data")


@dataclass(frozen=True, slots=True, kw_only=True)
class CandidateAnswer:
    text: str
    evidence_paths: tuple[RelPath, ...] = ()

    def __post_init__(self):
        _string(self.text, "text")
        _require("\x00" not in self.text, "candidate contains NUL")
        try:
            self.text.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise ContractError("candidate must be UTF-8 encodable") from exc
        _paths(self.evidence_paths, "evidence_paths", unique=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class StopAttempt:
    status: Literal["no_progress", "unsupported", "blocked_by_evidence"]
    reason: str

    def __post_init__(self):
        _enum(self.status, ("no_progress", "unsupported", "blocked_by_evidence"), "stop status")
        _string(self.reason, "reason", nonempty=True)


SolverStep: TypeAlias = ToolCall | CandidateAnswer | StopAttempt


@dataclass(frozen=True, slots=True, kw_only=True)
class StepRecord:
    step_index: int
    call: ToolCall
    result: ToolResult

    def __post_init__(self):
        _int(self.step_index, "step_index")
        _instance(self.call, ToolCall, "call")
        _instance(self.result, ToolResult, "result")
        _require(self.call.tool_name == self.result.tool_name, "step tool_name mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class FileChange:
    path: RelPath
    operation: Literal["write", "delete"]
    sha256: str | None = None

    def __post_init__(self):
        _rel(self.path)
        _enum(self.operation, ("write", "delete"), "operation")
        if self.operation == "write":
            _sha(self.sha256)
        else:
            _require(self.sha256 is None, "delete cannot have content hash")


@dataclass(frozen=True, slots=True, kw_only=True)
class AttemptResult:
    attempt_id: Id
    task_id: Id
    solver_id: Id
    status: Literal[
        "candidate",
        "no_progress",
        "unsupported",
        "blocked_by_evidence",
        "budget_exhausted",
        "deadline",
        "tool_error",
        "provider_error",
        "cancelled",
        "internal_error",
    ]
    workspace_root: Path
    candidate: CandidateAnswer | None = None
    changes: tuple[FileChange, ...] = ()
    snapshot_sha256: str | None = None
    steps_used: int = 0
    usage_request_ids: tuple[Id, ...] = ()
    error: ErrorInfo | None = None
    reason: str = ""

    def __post_init__(self):
        for name in ("attempt_id", "task_id", "solver_id"):
            _id(getattr(self, name), name)
        _enum(
            self.status,
            (
                "candidate",
                "no_progress",
                "unsupported",
                "blocked_by_evidence",
                "budget_exhausted",
                "deadline",
                "tool_error",
                "provider_error",
                "cancelled",
                "internal_error",
            ),
            "attempt status",
        )
        _absolute(self.workspace_root, "workspace_root")
        _int(self.steps_used, "steps_used")
        _ids(self.usage_request_ids, "usage_request_ids")
        _items(self.changes, FileChange, "changes")
        _unique(tuple(change.path for change in self.changes), "change paths")
        if self.error is not None or self.status in (
            "tool_error",
            "provider_error",
            "internal_error",
        ):
            _instance(self.error, ErrorInfo, "error")
        _string(self.reason, "reason", nonempty=self.status != "candidate")
        if self.status == "candidate":
            _instance(self.candidate, CandidateAnswer, "candidate")
            _sha(self.snapshot_sha256, "snapshot_sha256")
        else:
            _require(
                self.candidate is None and self.snapshot_sha256 is None and not self.changes,
                "non-candidate cannot have publication data",
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CheckResult:
    check_id: str
    status: Literal["passed", "failed", "inconclusive"]
    blocking: bool
    message: str

    def __post_init__(self):
        _string(self.check_id, "check_id")
        _enum(self.status, ("passed", "failed", "inconclusive"), "check status")
        _bool(self.blocking, "blocking")
        _string(self.message, "message")


@dataclass(frozen=True, slots=True, kw_only=True)
class ValidationResult:
    attempt_id: Id
    snapshot_sha256: str
    checks: tuple[CheckResult, ...]
    status: Literal["passed", "failed", "inconclusive"]

    def __post_init__(self):
        _id(self.attempt_id, "attempt_id")
        _sha(self.snapshot_sha256, "snapshot_sha256")
        _items(self.checks, CheckResult, "checks")
        _require(bool(self.checks), "checks must not be empty")
        _unique(tuple(check.check_id for check in self.checks), "check_id")
        _enum(self.status, ("passed", "failed", "inconclusive"), "validation status")
        statuses = {check.status for check in self.checks}
        aggregate = (
            "failed"
            if "failed" in statuses
            else ("inconclusive" if "inconclusive" in statuses else "passed")
        )
        _require(self.status == aggregate, "validation status disagrees with checks")


@dataclass(frozen=True, slots=True, kw_only=True)
class AcceptanceDecision:
    attempt_id: Id
    snapshot_sha256: str
    status: Literal["accept", "reject"]
    reason: str

    def __post_init__(self):
        _id(self.attempt_id, "attempt_id")
        _sha(self.snapshot_sha256, "snapshot_sha256")
        _enum(self.status, ("accept", "reject"), "acceptance status")
        _string(self.reason, "reason", nonempty=True)


@dataclass(frozen=True, slots=True, kw_only=True)
class PublicationPlan:
    answer_path: Path

    def __post_init__(self):
        _absolute(self.answer_path, "answer_path")


@dataclass(frozen=True, slots=True, kw_only=True)
class SubmissionReceipt:
    task_id: Id
    attempt_id: Id
    status: Literal["saved", "already_saved", "failed", "conflict"]
    answer_path: Path | None = None
    answer_sha256: str | None = None
    error: ErrorInfo | None = None

    def __post_init__(self):
        _id(self.task_id, "task_id")
        _id(self.attempt_id, "attempt_id")
        _enum(self.status, ("saved", "already_saved", "failed", "conflict"), "receipt status")
        if self.answer_path is not None:
            _absolute(self.answer_path, "answer_path")
        if self.answer_sha256 is not None:
            _sha(self.answer_sha256, "answer_sha256")
        if self.status in ("saved", "already_saved"):
            _require(
                self.answer_path is not None and self.answer_sha256 is not None,
                "saved receipt requires path and hash",
            )
            _require(self.error is None, "saved receipt cannot contain error")
        else:
            _instance(self.error, ErrorInfo, "error")


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadIssue:
    task_root: Path
    error: ErrorInfo

    def __post_init__(self):
        _absolute(self.task_root, "task_root")
        _instance(self.error, ErrorInfo, "error")


@dataclass(frozen=True, slots=True, kw_only=True)
class LoadReport:
    tasks: tuple[TaskSpec, ...]
    issues: tuple[LoadIssue, ...]

    def __post_init__(self):
        _items(self.tasks, TaskSpec, "tasks")
        _unique(tuple(task.task_id for task in self.tasks), "task_id")
        _items(self.issues, LoadIssue, "issues")


@dataclass(frozen=True, slots=True, kw_only=True)
class TaskProgress:
    task_id: Id
    status: Literal["pending", "running", "submitted", "deferred", "exhausted"]
    attempted_solvers: tuple[str, ...] = ()
    last_result: AttemptResult | None = None
    not_before_monotonic: float = 0.0

    def __post_init__(self):
        _id(self.task_id, "task_id")
        _enum(
            self.status,
            ("pending", "running", "submitted", "deferred", "exhausted"),
            "progress status",
        )
        _ids(self.attempted_solvers, "attempted_solvers", unique=False)
        _number(self.not_before_monotonic, "not_before_monotonic", minimum=0)
        if self.last_result is not None:
            _instance(self.last_result, AttemptResult, "last_result")
            _require(self.task_id == self.last_result.task_id, "last_result task_id mismatch")


@dataclass(frozen=True, slots=True, kw_only=True)
class PlanDecision:
    task_id: Id
    solver_id: Id
    budget: AttemptBudget

    def __post_init__(self):
        _id(self.task_id, "task_id")
        _id(self.solver_id, "solver_id")
        _instance(self.budget, AttemptBudget, "budget")


@dataclass(frozen=True, slots=True, kw_only=True)
class AttemptLimits:
    max_cost_units: Units
    max_llm_calls: int
    max_steps: int
    max_duration_s: float
    allowed_models: tuple[ModelId, ...]
    tool_limits: ToolLimits
    max_attempts_per_task: int = 2

    def __post_init__(self):
        _attempt_limits(self)
        _number(self.max_duration_s, "max_duration_s", positive=True)
        _int(self.max_attempts_per_task, "max_attempts_per_task", 1)


@dataclass(frozen=True, slots=True, kw_only=True)
class RunConfig:
    total_budget_units: Units
    deadline_monotonic: float
    attempt_limits: AttemptLimits
    mode: Literal["auto", "coverage", "portfolio"] = "auto"
    max_parallel_tasks: int = 2
    accounting_mode: Literal["strict", "best_effort"] = "strict"

    def __post_init__(self):
        _int(self.total_budget_units, "total_budget_units")
        _number(self.deadline_monotonic, "deadline_monotonic")
        _instance(self.attempt_limits, AttemptLimits, "attempt_limits")
        _enum(self.mode, ("auto", "coverage", "portfolio"), "mode")
        _int(self.max_parallel_tasks, "max_parallel_tasks", 1)
        _require(self.max_parallel_tasks <= 5, "max_parallel_tasks exceeds 5")
        _enum(self.accounting_mode, ("strict", "best_effort"), "accounting_mode")
