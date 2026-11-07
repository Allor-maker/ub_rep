"""Private attempt copies and verified snapshots; no execution or expected access."""

import asyncio
import hashlib
import json
import os
import stat
import tempfile
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable, Iterator, TypeVar

from nsu_budget_agent.contracts.models import (
    ArtifactRef,
    AttemptResult,
    CandidateAnswer,
    ContractError,
    ErrorInfo,
    FileChange,
    OperationFailed,
    TaskSpec,
)
from nsu_budget_agent.task_io._files import Directory, excluded, open_directory
from nsu_budget_agent.task_io.source_guard import source_operation

T = TypeVar("T")


async def joined_io(operation: Callable[..., T], *args, **kwargs) -> T:
    """Finish owned local I/O before cancellation permits staging cleanup."""
    worker = asyncio.create_task(asyncio.to_thread(operation, *args, **kwargs))
    cancelled = False
    while True:
        try:
            value = await asyncio.shield(worker)
            break
        except asyncio.CancelledError:
            cancelled = True
            if worker.done():
                worker.result()
                raise
    if cancelled:
        raise asyncio.CancelledError
    return value


def failure(code: str, message: str) -> OperationFailed:
    return OperationFailed(ErrorInfo(code=code, message=message))


@contextmanager
def work_directory(
    task: TaskSpec, *, expected: os.stat_result | None = None
) -> Iterator[Directory]:
    if ".." in task.work_root.parts or not task.work_root.is_relative_to(task.task_root):
        raise ContractError("unsafe work root")
    with open_directory(task.task_root, expected=expected) as root, ExitStack() as stack:
        current = root
        for part in task.work_root.relative_to(task.task_root).parts:
            if excluded(part):
                raise ContractError("excluded work root")
            current = stack.enter_context(current.child(part))
        yield current


@contextmanager
def artifact_file(root: Directory, path: PurePosixPath) -> Iterator[int]:
    with ExitStack() as stack:
        current = root
        for part in path.parts[:-1]:
            if excluded(part):
                raise ContractError("excluded artifact")
            current = stack.enter_context(current.child(part))
        if excluded(path.name):
            raise ContractError("excluded artifact")
        yield stack.enter_context(current.file(path.name))


def manifest(
    root: Path,
    protected: frozenset[PurePosixPath],
    *,
    max_bytes: int,
    source: Directory | None = None,
    skip_reserved: bool = False,
) -> tuple[ArtifactRef, ...]:
    result: list[ArtifactRef] = []
    total = 0

    def visit(directory: Directory, prefix: tuple[str, ...]) -> None:
        nonlocal total
        if len(prefix) > 64:
            raise failure("resource_limit", "workspace nesting limit")
        for name in directory.names():
            if excluded(name):
                if skip_reserved:
                    continue
                raise failure("denied", "reserved file in staging")
            info = directory.info(name)
            parts = (*prefix, name)
            if stat.S_ISDIR(info.st_mode):
                with directory.child(name, expected=info) as child:
                    visit(child, parts)
                continue
            if len(result) >= 10_000 or total + info.st_size > max_bytes:
                raise failure("resource_limit", "workspace size limit")
            digest = hashlib.sha256()
            size = 0
            with directory.file(name, expected=info) as descriptor:
                while chunk := os.read(descriptor, 65_536):
                    size += len(chunk)
                    if total + size > max_bytes:
                        raise failure("resource_limit", "workspace size limit")
                    digest.update(chunk)
            total += size
            path = PurePosixPath(*parts)
            result.append(
                ArtifactRef(
                    path=path,
                    size_bytes=size,
                    sha256=digest.hexdigest(),
                    protected=path in protected,
                )
            )

    if source is not None:
        visit(source, ())
    else:
        with open_directory(root) as directory:
            visit(directory, ())
    return tuple(sorted(result, key=lambda artifact: artifact.path.as_posix()))


def snapshot(candidate: CandidateAnswer, artifacts: tuple[ArtifactRef, ...]) -> str:
    body = [
        candidate.text,
        [[a.path.as_posix(), a.size_bytes, a.sha256, a.protected] for a in artifacts],
    ]
    return hashlib.sha256(
        json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


@dataclass
class Workspace:
    temporary: tempfile.TemporaryDirectory[str]
    task: TaskSpec
    source_identity: os.stat_result
    protected: set[PurePosixPath]
    protected_hashes: dict[PurePosixPath, str]
    frozen: tuple[ArtifactRef, ...] | None = None
    validation_temporary: tempfile.TemporaryDirectory[str] | None = None

    @property
    def root(self) -> Path:
        return Path(self.temporary.name)


class WorkspaceStore:
    def __init__(self, *, parent: Path | None = None, max_bytes: int = 64 * 1024 * 1024) -> None:
        self.parent = parent
        self.max_bytes = max_bytes
        self.records: dict[str, Workspace] = {}

    def create(self, task: TaskSpec, attempt_id: str) -> Path:
        temporary = tempfile.TemporaryDirectory(prefix="nsu-attempt-", dir=self.parent)
        root = Path(temporary.name).absolute()
        protected = {a.path for a in task.artifacts if a.protected}
        identity = task.task_root.lstat()
        record = Workspace(
            temporary,
            task,
            identity,
            protected,
            {a.path: a.sha256 for a in task.artifacts if a.protected},
        )
        try:
            if sum(a.size_bytes for a in task.artifacts) > self.max_bytes:
                raise failure("resource_limit", "source size limit")
            with source_operation(), work_directory(task, expected=identity) as source:
                for artifact in task.artifacts:
                    target = root.joinpath(*artifact.path.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    digest, size = hashlib.sha256(), 0
                    with (
                        artifact_file(source, artifact.path) as descriptor,
                        target.open("xb") as output,
                    ):
                        while chunk := os.read(descriptor, 65_536):
                            size += len(chunk)
                            if size > artifact.size_bytes:
                                raise failure("denied", "source changed")
                            digest.update(chunk)
                            output.write(chunk)
                    if size != artifact.size_bytes or digest.hexdigest() != artifact.sha256:
                        raise failure("denied", "source changed")
            self.records[attempt_id] = record
            return root
        except BaseException:
            temporary.cleanup()
            raise

    def protect(self, attempt_id: str, paths: tuple[PurePosixPath, ...]) -> None:
        record = self.records[attempt_id]
        original = {a.path: a.sha256 for a in record.task.artifacts}
        current = {
            a.path: a.sha256
            for a in manifest(record.root, frozenset(record.protected), max_bytes=self.max_bytes)
        }
        for path in paths:
            if path not in current or (path in original and current[path] != original[path]):
                raise failure("denied", "protected file changed before protection")
            record.protected.add(path)
            record.protected_hashes.setdefault(path, current[path])

    def freeze(
        self, attempt_id: str, candidate: CandidateAnswer
    ) -> tuple[tuple[FileChange, ...], str]:
        record = self.records[attempt_id]
        current = manifest(record.root, frozenset(record.protected), max_bytes=self.max_bytes)
        indexed = {a.path: a for a in current}
        for path, sha in record.protected_hashes.items():
            if path not in indexed or indexed[path].sha256 != sha:
                raise failure("denied", "protected file modified")
        if any(path not in indexed for path in candidate.evidence_paths):
            raise failure("invalid_output", "candidate evidence missing")
        before = {a.path: a for a in record.task.artifacts}
        changes = []
        for path in sorted(set(before) | set(indexed)):
            if path not in indexed:
                changes.append(FileChange(path=path, operation="delete"))
            elif path not in before or indexed[path].sha256 != before[path].sha256:
                changes.append(
                    FileChange(path=path, operation="write", sha256=indexed[path].sha256)
                )
        record.frozen = current
        for artifact in current:
            record.root.joinpath(*artifact.path.parts).chmod(0o444)
        return tuple(changes), snapshot(candidate, current)

    def verify(self, result: AttemptResult) -> None:
        record = self.records[result.attempt_id]
        current = manifest(record.root, frozenset(record.protected), max_bytes=self.max_bytes)
        if (
            result.workspace_root != record.root
            or current != record.frozen
            or result.candidate is None
        ):
            raise failure("denied", "staging changed after freeze")
        if snapshot(result.candidate, current) != result.snapshot_sha256:
            raise failure("denied", "snapshot mismatch")

    def cleanup(self, attempt_id: str) -> None:
        record = self.records.pop(attempt_id, None)
        if record is not None:
            if record.validation_temporary is not None:
                record.validation_temporary.cleanup()
            # Only our owned private directory; TemporaryDirectory handles read-only files.
            record.temporary.cleanup()

    def validation_copy(self, result: AttemptResult) -> tempfile.TemporaryDirectory[str]:
        self.verify(result)
        record = self.records[result.attempt_id]
        temporary = tempfile.TemporaryDirectory(prefix="nsu-validation-", dir=self.parent)
        try:
            with open_directory(record.root) as source:
                for artifact in record.frozen or ():
                    target = Path(temporary.name).joinpath(*artifact.path.parts)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with (
                        artifact_file(source, artifact.path) as descriptor,
                        target.open("xb") as output,
                    ):
                        while chunk := os.read(descriptor, 65_536):
                            output.write(chunk)
            copied = manifest(
                Path(temporary.name), frozenset(record.protected), max_bytes=self.max_bytes
            )
            if copied != record.frozen:
                raise failure("denied", "validation copy changed")
            record.validation_temporary = temporary
            return temporary
        except BaseException:
            temporary.cleanup()
            raise
