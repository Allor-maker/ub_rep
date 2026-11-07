"""Snapshot/acceptance/publication guards independent of SDK execution."""

import asyncio
import dataclasses
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from nsu_budget_agent.contracts.models import (
    AttemptResult,
    CandidateAnswer,
    CheckResult,
    FileChange,
    OperationFailed,
    ValidationResult,
)
from nsu_budget_agent.submission import AnswerOnlyPublisher, V1AcceptancePolicy
from nsu_budget_agent.task_io import OrganizerTaskAdapter
from nsu_budget_agent.task_io.source_guard import source_operation
from nsu_budget_agent.task_io.workspace import WorkspaceStore, joined_io


class SubmissionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    async def fixture(self):
        task_root = self.root / "task"
        task_root.mkdir()
        (task_root / "task.json").write_text(
            json.dumps({"task_id": "task", "instruction": "public", "reward": 1}), encoding="utf-8"
        )
        (task_root / "input.txt").write_bytes(b"source")
        task = await OrganizerTaskAdapter().load_task(task_root)
        store = WorkspaceStore(parent=self.root)
        root = store.create(task, "attempt")
        self.addCleanup(store.cleanup, "attempt")
        candidate = CandidateAnswer(text="  answer\n")
        changes, sha = store.freeze("attempt", candidate)
        result = AttemptResult(
            task_id=task.task_id,
            solver_id="generic",
            attempt_id="attempt",
            workspace_root=root,
            status="candidate",
            candidate=candidate,
            changes=changes,
            snapshot_sha256=sha,
        )
        validation = ValidationResult(
            attempt_id="attempt",
            snapshot_sha256=sha,
            checks=tuple(
                CheckResult(check_id=name, status="passed", blocking=True, message="fixture")
                for name in ("format", "paths", "protected_files", "snapshot")
            ),
            status="passed",
        )
        return task, store, result, validation

    async def test_policy_identity_missing_nonblocking_and_failed_checks(self):
        _, _, result, validation = await self.fixture()
        policy = V1AcceptancePolicy()
        self.assertEqual(policy.decide(result, validation).status, "accept")
        cases = [
            dataclasses.replace(validation, attempt_id="other"),
            dataclasses.replace(validation, snapshot_sha256="f" * 64),
            dataclasses.replace(validation, checks=validation.checks[:-1]),
            dataclasses.replace(
                validation,
                checks=(
                    dataclasses.replace(validation.checks[0], blocking=False),
                    *validation.checks[1:],
                ),
            ),
            dataclasses.replace(
                validation,
                checks=(
                    dataclasses.replace(validation.checks[0], status="failed"),
                    *validation.checks[1:],
                ),
                status="failed",
            ),
        ]
        for value in cases:
            self.assertEqual(policy.decide(result, value).status, "reject")

    async def test_atomic_publication_idempotence_and_conflict(self):
        task, store, result, validation = await self.fixture()
        publisher = AnswerOnlyPublisher(store)
        decision = V1AcceptancePolicy().decide(result, validation)
        receipts = await asyncio.gather(
            *(publisher.publish(task, result, decision) for _ in range(2))
        )
        self.assertEqual([r.status for r in receipts], ["saved", "already_saved"])
        self.assertEqual((task.task_root / "answer.txt").read_bytes(), b"  answer\n")
        self.assertEqual(receipts[0].answer_sha256, hashlib.sha256(b"  answer\n").hexdigest())
        self.assertFalse(list(task.task_root.glob(".nsu-answer-*")))
        other = dataclasses.replace(result, attempt_id="other")
        changed = dataclasses.replace(decision, attempt_id="other")
        self.assertEqual((await publisher.publish(task, other, changed)).status, "conflict")

    async def test_already_saved_rejects_different_exact_candidate_bytes(self):
        task, store, result, validation = await self.fixture()
        publisher = AnswerOnlyPublisher(store)
        decision = V1AcceptancePolicy().decide(result, validation)
        self.assertEqual((await publisher.publish(task, result, decision)).status, "saved")
        for text in ("answer\n", "  answer", "  answer\r\n", "другой ответ"):
            changed = dataclasses.replace(result, candidate=CandidateAnswer(text=text))
            self.assertEqual((await publisher.publish(task, changed, decision)).status, "conflict")
        self.assertEqual((task.task_root / "answer.txt").read_bytes(), b"  answer\n")
        self.assertEqual((await publisher.publish(task, result, decision)).status, "already_saved")

    async def test_source_changes_during_answer_preparation_refuse_commit(self):
        task, store, result, validation = await self.fixture()
        original = task.work_root / "input.txt"
        added = task.work_root / "new.txt"
        real_fsync = os.fsync
        for change in ("modify", "delete", "add"):
            with self.subTest(change=change):
                mutated = False

                def fsync_then_change(descriptor):
                    nonlocal mutated
                    real_fsync(descriptor)
                    if not mutated:
                        mutated = True
                        if change == "modify":
                            original.write_bytes(b"changed")
                        elif change == "delete":
                            original.unlink()
                        else:
                            added.write_bytes(b"new")

                try:
                    with patch(
                        "nsu_budget_agent.submission.publisher.os.fsync",
                        side_effect=fsync_then_change,
                    ):
                        receipt = await AnswerOnlyPublisher(store).publish(
                            task, result, V1AcceptancePolicy().decide(result, validation)
                        )
                    self.assertTrue(mutated)
                    self.assertEqual(receipt.status, "conflict")
                    self.assertFalse((task.task_root / "answer.txt").exists())
                    self.assertFalse(list(task.task_root.glob(".nsu-answer-*")))
                finally:
                    original.write_bytes(b"source")
                    added.unlink(missing_ok=True)

    async def test_staging_rechecked_after_answer_preparation(self):
        task, store, result, validation = await self.fixture()
        real_fsync = os.fsync

        def fsync_then_change(descriptor):
            real_fsync(descriptor)
            path = result.workspace_root / "input.txt"
            path.chmod(0o600)
            path.write_bytes(b"changed")

        with patch("nsu_budget_agent.submission.publisher.os.fsync", side_effect=fsync_then_change):
            receipt = await AnswerOnlyPublisher(store).publish(
                task, result, V1AcceptancePolicy().decide(result, validation)
            )
        self.assertEqual(receipt.status, "failed")
        self.assertFalse((task.task_root / "answer.txt").exists())
        self.assertFalse(list(task.task_root.glob(".nsu-answer-*")))

    async def test_agent_source_writer_finishes_before_publication_checks(self):
        task, store, result, validation = await self.fixture()
        started, release, requested = threading.Event(), threading.Event(), threading.Event()
        publisher = AnswerOnlyPublisher(store)
        original = publisher._publish

        def writer():
            with source_operation():
                (task.work_root / "input.txt").write_bytes(b"changed")
                started.set()
                if not release.wait(5):
                    raise TimeoutError("test synchronization")

        def publish(task, result):
            requested.set()
            return original(task, result)

        worker = asyncio.create_task(joined_io(writer))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        try:
            with patch.object(publisher, "_publish", side_effect=publish):
                publication = asyncio.create_task(
                    publisher.publish(task, result, V1AcceptancePolicy().decide(result, validation))
                )
                self.assertTrue(await asyncio.to_thread(requested.wait, 2))
                self.assertFalse(publication.done())
                release.set()
                receipt = await publication
        finally:
            release.set()
            await worker
        self.assertEqual(receipt.status, "conflict")
        self.assertFalse((task.task_root / "answer.txt").exists())

    async def test_agent_source_operation_cannot_interleave_precommit(self):
        task, store, result, validation = await self.fixture()
        preparing, release, requested, entered = (threading.Event() for _ in range(4))
        real_fsync = os.fsync

        def fsync_and_pause(descriptor):
            real_fsync(descriptor)
            if not preparing.is_set():
                preparing.set()
                if not release.wait(5):
                    raise TimeoutError("test synchronization")

        def agent_operation():
            requested.set()
            with source_operation():
                entered.set()
                # A future source writer must additionally honor submitted terminal state.
                return (task.task_root / "answer.txt").read_bytes()

        with patch("nsu_budget_agent.submission.publisher.os.fsync", side_effect=fsync_and_pause):
            publication = asyncio.create_task(
                AnswerOnlyPublisher(store).publish(
                    task, result, V1AcceptancePolicy().decide(result, validation)
                )
            )
            try:
                self.assertTrue(await asyncio.to_thread(preparing.wait, 2))
                worker = asyncio.create_task(joined_io(agent_operation))
                self.assertTrue(await asyncio.to_thread(requested.wait, 2))
                self.assertFalse(entered.is_set())
            finally:
                release.set()
                receipt = await publication
            self.assertEqual(await worker, b"  answer\n")
        self.assertEqual(receipt.status, "saved")

    async def test_staging_mutation_rejected_before_publication(self):
        task, store, result, validation = await self.fixture()
        path = result.workspace_root / "input.txt"
        path.chmod(0o600)
        path.write_bytes(b"changed")
        receipt = await AnswerOnlyPublisher(store).publish(
            task, result, V1AcceptancePolicy().decide(result, validation)
        )
        self.assertEqual(receipt.status, "failed")
        self.assertFalse((task.task_root / "answer.txt").exists())

    async def test_original_mutation_rejected(self):
        task, store, result, validation = await self.fixture()
        (task.work_root / "input.txt").write_bytes(b"changed")
        receipt = await AnswerOnlyPublisher(store).publish(
            task, result, V1AcceptancePolicy().decide(result, validation)
        )
        self.assertEqual(receipt.status, "conflict")
        self.assertFalse((task.task_root / "answer.txt").exists())

    async def test_source_changes_explicitly_unsupported(self):
        task, store, result, validation = await self.fixture()
        result = dataclasses.replace(
            result, changes=(FileChange(path=PurePosixPath("input.txt"), operation="delete"),)
        )
        receipt = await AnswerOnlyPublisher(store).publish(
            task, result, V1AcceptancePolicy().decide(result, validation)
        )
        self.assertEqual(receipt.error.code, "unsupported")
        self.assertEqual((task.work_root / "input.txt").read_bytes(), b"source")
        self.assertFalse((task.task_root / "answer.txt").exists())
        path = result.workspace_root / "input.txt"
        path.chmod(0o600)
        path.write_bytes(b"changed")
        _, sha = store.freeze("attempt", result.candidate)
        forged = dataclasses.replace(result, changes=(), snapshot_sha256=sha)
        validation = dataclasses.replace(validation, snapshot_sha256=sha)
        receipt = await AnswerOnlyPublisher(store).publish(
            task, forged, V1AcceptancePolicy().decide(forged, validation)
        )
        self.assertEqual(receipt.error.code, "unsupported")
        self.assertFalse((task.task_root / "answer.txt").exists())

    async def test_hardlink_failure_leaves_no_answer_or_temporary(self):
        task, store, result, validation = await self.fixture()
        with patch("nsu_budget_agent.submission.publisher.os.link", side_effect=OSError("fixture")):
            receipt = await AnswerOnlyPublisher(store).publish(
                task, result, V1AcceptancePolicy().decide(result, validation)
            )
        self.assertEqual(receipt.status, "failed")
        self.assertFalse((task.task_root / "answer.txt").exists())
        self.assertFalse(list(task.task_root.glob(".nsu-answer-*")))

    async def test_cleanup_failure_after_commit_still_reports_saved(self):
        task, store, result, validation = await self.fixture()
        with patch("nsu_budget_agent.submission.publisher._unlink", side_effect=OSError("fixture")):
            receipt = await AnswerOnlyPublisher(store).publish(
                task, result, V1AcceptancePolicy().decide(result, validation)
            )
        self.assertEqual(receipt.status, "saved")
        self.assertTrue((task.task_root / "answer.txt").exists())

    async def test_reserved_files_and_missing_evidence_rejected(self):
        task, store, result, _ = await self.fixture()
        (result.workspace_root / "expected.txt").write_text("forbidden", encoding="utf-8")
        with self.assertRaises(OperationFailed):
            store.verify(result)
        (result.workspace_root / "expected.txt").unlink()
        with self.assertRaises(OperationFailed):
            store.freeze(
                "attempt", CandidateAnswer(text="x", evidence_paths=(PurePosixPath("absent"),))
            )

    async def test_repeated_cancellation_joins_owned_io(self):
        started, release, finished = threading.Event(), threading.Event(), threading.Event()

        def worker():
            started.set()
            release.wait(5)
            finished.set()

        operation = asyncio.create_task(joined_io(worker))
        await asyncio.to_thread(started.wait, 2)
        operation.cancel()
        await asyncio.sleep(0)
        operation.cancel()
        release.set()
        with self.assertRaises(asyncio.CancelledError):
            await operation
        self.assertTrue(finished.is_set())

    async def test_protection_cannot_be_added_after_modification(self):
        _, store, result, _ = await self.fixture()
        path = result.workspace_root / "input.txt"
        path.chmod(0o600)
        path.write_bytes(b"changed")
        with self.assertRaises(OperationFailed):
            store.protect("attempt", (PurePosixPath("input.txt"),))
