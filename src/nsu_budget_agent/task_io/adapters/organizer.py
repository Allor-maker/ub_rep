"""Organizer task.json adapter. Never expose raw manifests or expected values."""

import asyncio
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath

from nsu_budget_agent.contracts.models import (
    ArtifactRef,
    ContractError,
    ErrorInfo,
    OperationFailed,
    TaskSpec,
    _int,
    parse_rel_path,
)

from .._files import Directory, excluded, open_directory


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def _constant(value: str) -> None:
    raise ValueError("nonfinite JSON")


def _text(data: dict[str, object], name: str, *, optional: bool = False) -> str | None:
    value = data.get(name)
    if value is None and optional:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ContractError("invalid public manifest field")
    return value


def _instruction_tests(instruction: str) -> frozenset[str]:
    """Recognize explicit test runners and conventional verification script names."""
    commands = re.findall(
        r"\b(pytest|unittest|python(?:3)?(?:\s+-m\s+(?:pytest|unittest))?)\s+"
        r"[`\"']?([\w./-]+\.py)\b",
        instruction,
    )
    paths: set[str] = set()
    for command, raw in commands:
        # Remove only explicit current-directory prefixes; reject traversal before normalization.
        while raw.startswith("./"):
            raw = raw[2:]
        try:
            relative = parse_rel_path(raw)
        except ContractError:
            continue
        name = PurePosixPath(relative).name.casefold()
        explicit_runner = command.startswith(("pytest", "unittest")) or " -m " in command
        if explicit_runner or name.startswith(("test_", "verify_")) or name.endswith("_test.py"):
            paths.add(relative.as_posix())
    return frozenset(paths)


class OrganizerTaskAdapter:
    """Limits are local input safeguards, not asserted competition limits."""

    def __init__(
        self,
        *,
        max_manifest_bytes: int = 1_048_576,
        max_artifact_bytes: int = 268_435_456,
        max_entries: int = 10_000,
        max_depth: int = 64,
    ) -> None:
        for value in (max_manifest_bytes, max_artifact_bytes, max_entries, max_depth):
            _int(value, "input limit", 1)
        self._manifest_limit = max_manifest_bytes
        self._artifact_limit = max_artifact_bytes
        self._entries_limit = max_entries
        self._depth_limit = max_depth

    async def load_task(self, task_root: Path) -> TaskSpec:
        # Worker only reads bounded input; it never calls the ledger or solvers.
        return await asyncio.to_thread(self._load, task_root)

    async def _load_checked(self, task_root: Path, expected: os.stat_result) -> TaskSpec:
        return await asyncio.to_thread(self._load, task_root, expected)

    def _load(self, task_root: Path, expected: os.stat_result | None = None) -> TaskSpec:
        try:
            with open_directory(task_root, expected=expected) as root:
                with root.file("task.json") as descriptor:
                    raw = bytearray()
                    while chunk := os.read(descriptor, min(65_536, self._manifest_limit + 1)):
                        raw.extend(chunk)
                        if len(raw) > self._manifest_limit:
                            raise OperationFailed(
                                ErrorInfo(code="resource_limit", message="manifest too large")
                            )
                data = json.loads(
                    raw.decode("utf-8-sig"), object_pairs_hook=_object, parse_constant=_constant
                )
                if not isinstance(data, dict):
                    raise ContractError("manifest must be an object")
                # Explicit allowlist: expected and arbitrary metadata are never accessed.
                task_id = _text(data, "task_id")
                instruction = _text(data, "instruction")
                reward = data.get("reward")
                if type(reward) is not int:
                    raise ContractError("reward must be an integer")
                _int(reward, "reward")
                type_hint = _text(data, "task_type", optional=True)
                alternate_type = _text(data, "type", optional=True)
                if (
                    type_hint is not None
                    and alternate_type is not None
                    and type_hint != alternate_type
                ):
                    raise ContractError("conflicting public type hints")
                check_hint = _text(data, "check", optional=True)
                names = root.names(limit=self._entries_limit)
                workspace_info = root.info("workspace") if "workspace" in names else None
                if workspace_info is not None and stat.S_ISDIR(workspace_info.st_mode):
                    with root.child("workspace", expected=workspace_info) as work:
                        artifacts = self._artifacts(work, instruction or "")
                        work_root = work.path
                else:
                    artifacts = self._artifacts(root, instruction or "")
                    work_root = root.path
                # Public fields have been validated above; DTO enforces the remaining invariants.
                assert task_id is not None and instruction is not None and isinstance(reward, int)
                return TaskSpec(
                    task_id=task_id,
                    instruction=instruction,
                    reward=reward,
                    task_root=root.path,
                    work_root=work_root,
                    artifacts=artifacts,
                    type_hint=type_hint if type_hint is not None else alternate_type,
                    check_hint=check_hint,
                )
        except OperationFailed:
            raise
        except (ValueError, UnicodeError, RecursionError):
            raise OperationFailed(
                ErrorInfo(code="invalid_input", message="invalid task manifest or file layout")
            ) from None
        except OSError:
            raise OperationFailed(
                ErrorInfo(code="io_error", message="cannot safely read task input")
            ) from None

    def _artifacts(self, root: Directory, instruction: str) -> tuple[ArtifactRef, ...]:
        artifacts: list[ArtifactRef] = []
        entries = total_bytes = 0
        command_files = _instruction_tests(instruction)

        def visit(directory: Directory, prefix: tuple[str, ...]) -> None:
            nonlocal entries, total_bytes
            if len(prefix) > self._depth_limit:
                raise OperationFailed(
                    ErrorInfo(code="resource_limit", message="artifact nesting limit exceeded")
                )
            names = directory.names(limit=self._entries_limit)
            if len({name.casefold() for name in names}) != len(names):
                raise ContractError("case-ambiguous file names")
            for name in names:
                entries += 1
                if entries > self._entries_limit:
                    raise OperationFailed(
                        ErrorInfo(code="resource_limit", message="artifact entry limit exceeded")
                    )
                if excluded(name):
                    continue
                info = directory.info(name)
                parts = (*prefix, name)
                if stat.S_ISDIR(info.st_mode):
                    with directory.child(name, expected=info) as child:
                        visit(child, parts)
                    continue
                if info.st_size + total_bytes > self._artifact_limit:
                    raise OperationFailed(
                        ErrorInfo(code="resource_limit", message="artifact byte limit exceeded")
                    )
                digest = hashlib.sha256()
                size = 0
                with directory.file(name, expected=info) as descriptor:
                    while chunk := os.read(descriptor, 65_536):
                        size += len(chunk)
                        if size + total_bytes > self._artifact_limit:
                            raise OperationFailed(
                                ErrorInfo(
                                    code="resource_limit", message="artifact byte limit exceeded"
                                )
                            )
                        digest.update(chunk)
                total_bytes += size
                # Conservative public test conventions; additional tests become protected in B1.
                protected = (
                    any(part.casefold() in ("tests", "test") for part in parts[:-1])
                    or (name.casefold().startswith("test_") or name.casefold().endswith("_test.py"))
                    or "/".join(parts) in command_files
                )
                artifacts.append(
                    ArtifactRef(
                        path=parse_rel_path("/".join(parts)),
                        size_bytes=size,
                        sha256=digest.hexdigest(),
                        protected=protected,
                    )
                )

        visit(root, ())
        return tuple(artifacts)
