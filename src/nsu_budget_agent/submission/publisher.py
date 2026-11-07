"""A5-lite: answer-only, atomic no-clobber publication with a per-task journal."""

import asyncio
import hashlib
import os
from contextlib import suppress
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from nsu_budget_agent.contracts.models import (
    AcceptanceDecision,
    AttemptResult,
    ContractError,
    ErrorInfo,
    OperationFailed,
    PublicationPlan,
    SubmissionReceipt,
    TaskSpec,
)
from nsu_budget_agent.task_io._files import Directory, open_directory
from nsu_budget_agent.task_io.source_guard import source_operation
from nsu_budget_agent.task_io.workspace import (
    WorkspaceStore,
    joined_io,
    manifest,
    work_directory,
)


class RootSubmissionLayout:
    def plan(self, task: TaskSpec) -> PublicationPlan:
        return PublicationPlan(answer_path=task.task_root / "answer.txt")


def _unlink(directory: Directory, name: str) -> None:
    if os.name == "nt":
        os.unlink(directory.path / name)
    else:
        os.unlink(name, dir_fd=directory.descriptor)


class AnswerOnlyPublisher:
    def __init__(self, workspaces: WorkspaceStore) -> None:
        self._workspaces = workspaces
        self._locks: dict[Path, asyncio.Lock] = {}
        self._journal: dict[Path, tuple[SubmissionReceipt, str]] = {}

    async def publish(
        self, task: TaskSpec, result: AttemptResult, decision: AcceptanceDecision
    ) -> SubmissionReceipt:
        if (
            result.task_id != task.task_id
            or result.status != "candidate"
            or decision.status != "accept"
            or decision.attempt_id != result.attempt_id
            or decision.snapshot_sha256 != result.snapshot_sha256
        ):
            raise ContractError("publication requires the matching accepted candidate")
        async with self._locks.setdefault(task.task_root, asyncio.Lock()):
            return await joined_io(self._publish, task, result)

    def receipt_for(self, task_root: Path) -> SubmissionReceipt | None:
        previous = self._journal.get(task_root)
        return previous[0] if previous is not None else None

    def _publish(self, task: TaskSpec, result: AttemptResult) -> SubmissionReceipt:
        # Shared across publisher instances; all trusted original-tree writers use it.
        with source_operation():
            return self._publish_locked(task, result)

    def _source_matches(self, task: TaskSpec, identity: os.stat_result) -> bool:
        try:
            with work_directory(task, expected=identity) as source:
                current = manifest(
                    task.work_root,
                    frozenset(a.path for a in task.artifacts if a.protected),
                    max_bytes=self._workspaces.max_bytes,
                    source=source,
                    skip_reserved=True,
                )
            return {a.path: a for a in current} == {a.path: a for a in task.artifacts}
        except (FileNotFoundError, ContractError):
            return False

    def _publish_locked(self, task: TaskSpec, result: AttemptResult) -> SubmissionReceipt:
        target = task.task_root / "answer.txt"
        assert result.candidate is not None and result.snapshot_sha256 is not None
        content = result.candidate.text.encode("utf-8")
        digest = hashlib.sha256(content).hexdigest()
        previous = self._journal.get(task.task_root)
        try:
            if previous is not None:
                receipt, accepted_snapshot = previous
                if (
                    receipt.attempt_id != result.attempt_id
                    or accepted_snapshot != result.snapshot_sha256
                    or receipt.answer_sha256 != digest
                ):
                    return self._failed(task, result, "conflict", "task already submitted")
                with (
                    open_directory(task.task_root) as directory,
                    directory.file("answer.txt") as descriptor,
                ):
                    actual = hashlib.sha256()
                    while chunk := os.read(descriptor, 65_536):
                        actual.update(chunk)
                if actual.hexdigest() != receipt.answer_sha256:
                    return self._failed(task, result, "conflict", "saved answer changed")
                return replace(receipt, status="already_saved")
            if result.changes:
                return self._failed(
                    task,
                    result,
                    "failed",
                    "A5-lite does not publish source changes",
                    code="unsupported",
                )
            self._workspaces.verify(result)
            record = self._workspaces.records[result.attempt_id]
            # Independently verify the claimed empty diff, including added/deleted files.
            original = {a.path: (a.size_bytes, a.sha256) for a in task.artifacts}
            frozen = {a.path: (a.size_bytes, a.sha256) for a in record.frozen or ()}
            if original != frozen:
                return self._failed(
                    task, result, "failed", "candidate requires source changes", code="unsupported"
                )
            with open_directory(task.task_root, expected=record.source_identity) as directory:
                if not self._source_matches(task, record.source_identity):
                    return self._failed(task, result, "conflict", "source changed")
                if "answer.txt" in directory.names():
                    return self._failed(task, result, "conflict", "answer already exists")
                temporary = ".nsu-answer-" + uuid4().hex + ".tmp"
                created = False
                committed = False
                try:
                    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    if os.name == "nt":
                        descriptor = os.open(directory.path / temporary, flags | os.O_BINARY, 0o600)
                    else:
                        descriptor = os.open(
                            temporary,
                            flags | getattr(os, "O_NOFOLLOW", 0),
                            0o600,
                            dir_fd=directory.descriptor,
                        )
                    created = True
                    with os.fdopen(descriptor, "wb") as output:
                        output.write(content)
                        output.flush()
                        os.fsync(output.fileno())
                    # Detect changes during preparation, while keeping our writers excluded.
                    self._workspaces.verify(result)
                    if not self._source_matches(task, record.source_identity):
                        return self._failed(
                            task, result, "conflict", "source changed before commit"
                        )
                    # Verify the root pathname still denotes our pinned directory before commit.
                    if not os.path.samestat(record.source_identity, task.task_root.lstat()):
                        return self._failed(task, result, "conflict", "task root changed")
                    if os.name == "nt":
                        os.link(directory.path / temporary, target)
                    else:
                        os.link(
                            temporary,
                            "answer.txt",
                            src_dir_fd=directory.descriptor,
                            dst_dir_fd=directory.descriptor,
                            follow_symlinks=False,
                        )
                    receipt = SubmissionReceipt(
                        task_id=task.task_id,
                        attempt_id=result.attempt_id,
                        status="saved",
                        answer_path=target,
                        answer_sha256=digest,
                    )
                    self._journal[task.task_root] = (receipt, result.snapshot_sha256)
                    committed = True
                    return receipt
                finally:
                    if created:
                        if committed:
                            # A cleanup failure cannot undo or misreport the committed answer.
                            with suppress(OSError):
                                _unlink(directory, temporary)
                        else:
                            _unlink(directory, temporary)
                    if committed and os.name != "nt":
                        with suppress(OSError):
                            os.fsync(directory.descriptor)
        except FileExistsError:
            return self._failed(task, result, "conflict", "answer appeared during publication")
        except (OSError, ValueError, OperationFailed):
            return self._failed(task, result, "failed", "safe publication failed")

    @staticmethod
    def _failed(
        task: TaskSpec, result: AttemptResult, status, message: str, *, code: str = "io_error"
    ) -> SubmissionReceipt:
        return SubmissionReceipt(
            task_id=task.task_id,
            attempt_id=result.attempt_id,
            status=status,
            error=ErrorInfo(code=code, message=message),
        )
