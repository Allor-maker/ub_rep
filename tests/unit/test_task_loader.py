"""A4 read-only input: synthetic tasks, no organizer expected or solver code."""

import asyncio
import dataclasses
import errno
import hashlib
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from nsu_budget_agent.contracts.models import OperationFailed, TaskSpec
from nsu_budget_agent.task_io import DirectoryTaskLoader, OrganizerTaskAdapter
from nsu_budget_agent.task_io._files import Directory


class TaskLoaderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def task(self, name="task", *, workspace=True, **fields):
        root = self.root / name
        root.mkdir()
        manifest = {
            "task_id": name,
            "instruction": "Read the available files.",
            "reward": 2,
            "task_type": "math",
            "check": "exact_int",
            "expected": {"secret": "ORACLE NEVER EXPOSED"},
            **fields,
        }
        (root / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        if workspace:
            (root / "workspace").mkdir()
        return root

    def symlink(self, link, target, *, directory=False):
        try:
            link.symlink_to(target, target_is_directory=directory)
        except OSError as exc:
            unavailable = getattr(exc, "winerror", None) == 1314 or (
                os.name == "posix" and exc.errno in (errno.EPERM, errno.ENOSYS, errno.ENOTSUP)
            )
            if unavailable:
                self.skipTest(f"symlink capability unavailable: {exc}")
            raise

    async def failure(self, root, code=None, adapter=None):
        with self.assertRaises(OperationFailed) as caught:
            await (adapter or OrganizerTaskAdapter()).load_task(root)
        if code is not None:
            self.assertEqual(caught.exception.error.code, code)

    async def test_four_types_and_unknown_or_absent_hint(self):
        for index, hint in enumerate(
            ("math", "search", "data_analysis", "coding", "new_type", None)
        ):
            root = self.task(str(index), task_type=hint)
            task = await OrganizerTaskAdapter().load_task(root)
            self.assertEqual(task.type_hint, hint)
            self.assertEqual(task.reward, 2)
            self.assertEqual(task.check_hint, "exact_int")

    async def test_type_alias_and_conflict(self):
        root = self.task(task_type=None, type="search")
        self.assertEqual((await OrganizerTaskAdapter().load_task(root)).type_hint, "search")
        manifest = json.loads((root / "task.json").read_text(encoding="utf-8"))
        manifest["task_type"] = "math"
        (root / "task.json").write_text(json.dumps(manifest), encoding="utf-8")
        await self.failure(root, "invalid_input")

    async def test_arbitrary_nested_artifacts_hashes_and_public_test_protection(self):
        root = self.task(instruction="Run python3 verify_custom.py; fix library.py.")
        work = root / "workspace"
        (work / "new-directory").mkdir()
        (work / "tests").mkdir()
        contents = {
            "new-directory/unseen.bin": b"\x00\xff\x01",
            "test_original.py": b"assert True\n",
            "tests/unusual_name.txt": b"protected\n",
            "verify_custom.py": b"assert True\n",
            "library.py": b"pass\n",
        }
        for name, data in contents.items():
            (work / name).write_bytes(data)
        task = await OrganizerTaskAdapter().load_task(root)
        self.assertEqual(task.task_root, root)
        self.assertEqual(task.work_root, work)
        artifacts = {str(item.path): item for item in task.artifacts}
        self.assertEqual(set(artifacts), set(contents))
        for name, data in contents.items():
            self.assertEqual(artifacts[name].size_bytes, len(data))
            self.assertEqual(artifacts[name].sha256, hashlib.sha256(data).hexdigest())
        for name in ("test_original.py", "tests/unusual_name.txt", "verify_custom.py"):
            self.assertTrue(artifacts[name].protected)
        self.assertFalse(artifacts["library.py"].protected)

    async def test_no_workspace_and_exclusions_never_opened(self):
        root = self.task(workspace=False)
        for name in ("expected", ".git", "__pycache__"):
            (root / name).mkdir()
            (root / name / "private").write_bytes(b"do not read")
        for name in ("answer.txt", "expected.json", "expected_private.txt"):
            (root / name).write_bytes(b"do not read")
        (root / "fresh-name.txt").write_bytes(b"public")
        opened = []
        original = Directory.file

        def guarded(directory, name, **kwargs):
            opened.append(name)
            self.assertIn(name, ("task.json", "fresh-name.txt"))
            return original(directory, name, **kwargs)

        with patch.object(Directory, "file", new=guarded):
            task = await OrganizerTaskAdapter().load_task(root)
        self.assertEqual(opened, ["task.json", "fresh-name.txt"])
        self.assertEqual(task.work_root, root)
        self.assertEqual([str(item.path) for item in task.artifacts], ["fresh-name.txt"])
        self.assertNotIn("ORACLE", repr(dataclasses.asdict(task)))
        self.assertEqual((root / "answer.txt").read_bytes(), b"do not read")

    async def test_workspace_file_falls_back_to_task_root(self):
        root = self.task(workspace=False)
        (root / "workspace").write_bytes(b"ordinary file")
        task = await OrganizerTaskAdapter().load_task(root)
        self.assertEqual(task.work_root, root)
        self.assertEqual(str(task.artifacts[0].path), "workspace")

    async def test_invalid_manifests_are_isolated_and_errors_sanitized(self):
        good = self.task("good")
        for index, fields in enumerate(
            (
                {"reward": True},
                {"reward": -1},
                {"reward": 1.5},
                {"instruction": ""},
                {"task_id": None},
                {"check": {}},
                {"task_type": []},
            )
        ):
            self.task(f"bad-{index}", **fields)
        report = await DirectoryTaskLoader().load(self.root)
        self.assertEqual([task.task_root for task in report.tasks], [good])
        self.assertEqual(len(report.issues), 7)
        self.assertNotIn("ORACLE", repr(report))

    async def test_malformed_json_duplicate_keys_and_missing_manifest(self):
        for index, raw in enumerate((b"{", b"[]", b'{"reward":1,"reward":2}', b'{"reward":NaN}')):
            root = self.task(str(index))
            (root / "task.json").write_bytes(raw)
            await self.failure(root, "invalid_input")
        empty = self.root / "empty"
        empty.mkdir()
        await self.failure(empty, "io_error")

    async def test_duplicate_ids_all_excluded_and_stable_discovery(self):
        self.task("z", task_id="same")
        self.task("a", task_id="same")
        self.task("unique")
        (self.root / ".service").mkdir()
        (self.root / "README").write_text("not a task", encoding="utf-8")
        report = await DirectoryTaskLoader().load(self.root)
        self.assertEqual([task.task_id for task in report.tasks], ["unique"])
        self.assertEqual([issue.task_root.name for issue in report.issues], ["a", "z"])

    async def test_extreme_rewards_keep_integer_precision(self):
        for index, reward in enumerate((0, 10**300)):
            task = await OrganizerTaskAdapter().load_task(self.task(str(index), reward=reward))
            self.assertEqual(task.reward, reward)

    async def test_input_limits(self):
        root = self.task()
        (root / "workspace" / "large.bin").write_bytes(b"12345")
        for adapter in (
            OrganizerTaskAdapter(max_manifest_bytes=10),
            OrganizerTaskAdapter(max_artifact_bytes=4),
        ):
            await self.failure(root, "resource_limit", adapter)
        (root / "workspace" / "other").write_bytes(b"")
        await self.failure(root, "resource_limit", OrganizerTaskAdapter(max_entries=1))
        (root / "workspace" / "a" / "b").mkdir(parents=True)
        await self.failure(root, "resource_limit", OrganizerTaskAdapter(max_depth=1))

    async def test_invalid_input_root_and_parent_traversal(self):
        for path in (self.root / "missing", self.root / ".." / self.root.name):
            with self.assertRaises(OperationFailed):
                await DirectoryTaskLoader().load(path)

    async def test_links_to_other_tasks_are_rejected(self):
        one, two = self.task("one"), self.task("two")
        target = two / "workspace" / "private.txt"
        target.write_text("OTHER TASK", encoding="utf-8")
        self.symlink(one / "workspace" / "escape", target)
        await self.failure(one)
        report = await DirectoryTaskLoader().load(self.root)
        self.assertEqual([task.task_id for task in report.tasks], ["two"])
        self.assertEqual(len(report.issues), 1)

    async def test_symlink_in_root_manifest_workspace_and_ancestor(self):
        target = self.task("target")
        link = self.root / "link"
        self.symlink(link, target, directory=True)
        await self.failure(link)
        await self.failure(link / "workspace")
        manifest_link = self.task("manifest")
        (manifest_link / "task.json").unlink()
        self.symlink(manifest_link / "task.json", target / "task.json")
        await self.failure(manifest_link)
        work_link = self.task("work", workspace=False)
        self.symlink(work_link / "workspace", target / "workspace", directory=True)
        await self.failure(work_link)

    async def test_symlink_helper_does_not_hide_unexpected_os_errors(self):
        with patch.object(Path, "symlink_to", side_effect=OSError(errno.ENOSPC, "disk full")):
            with self.assertRaises(OSError) as caught:
                self.symlink(self.root / "link", self.root / "target")
        self.assertEqual(caught.exception.errno, errno.ENOSPC)

    async def test_directory_swap_to_expected_is_rejected_before_reading(self):
        for swap_workspace in (False, True):
            with self.subTest(workspace=swap_workspace):
                root = self.task("workspace_case" if swap_workspace else "nested_case")
                target = root / "workspace"
                if not swap_workspace:
                    target = target / "data"
                    target.mkdir()
                (target / "public.txt").write_bytes(b"public")
                expected = root / "expected"
                expected.mkdir()
                (expected / "secret.txt").write_bytes(b"SECRET_EXPECTED")
                swapped = []
                original_file = Directory.file
                opened = []

                def track_file(directory, name, **kwargs):
                    opened.append(name)
                    return original_file(directory, name, **kwargs)

                def replace_directory():
                    target.rename(root / "moved_original")
                    expected.rename(target)
                    swapped.append(True)

                if os.name == "nt":
                    from nsu_budget_agent.task_io import _windows

                    original_open = _windows.open_handle

                    def intercept(path, *, directory):
                        if Path(path) == target and directory and not swapped:
                            replace_directory()
                        return original_open(path, directory=directory)

                    opener = patch.object(_windows, "open_handle", new=intercept)
                else:
                    original_os_open = os.open

                    def intercept(path, flags, *args, **kwargs):
                        if path == target.name and kwargs.get("dir_fd") is not None and not swapped:
                            replace_directory()
                        return original_os_open(path, flags, *args, **kwargs)

                    opener = patch("nsu_budget_agent.task_io._files.os.open", new=intercept)
                with opener, patch.object(Directory, "file", new=track_file):
                    await self.failure(root)
                self.assertTrue(swapped)
                self.assertNotIn("secret.txt", opened)

    async def test_file_swap_to_expected_after_listing_is_rejected(self):
        root = self.task()
        file = root / "workspace" / "public.txt"
        file.write_bytes(b"public")
        expected = root / "expected"
        expected.mkdir()
        hidden = expected / "hidden.txt"
        hidden.write_bytes(b"SECRET_EXPECTED")
        original = Directory.file
        swapped = []

        def intercept(directory, name, **kwargs):
            if name == "public.txt":
                file.rename(root / "moved_public")
                hidden.rename(file)
                swapped.append(True)
            return original(directory, name, **kwargs)

        with patch.object(Directory, "file", new=intercept):
            await self.failure(root)
        self.assertTrue(swapped)

    async def test_directory_swap_after_listing_before_child_call_is_rejected(self):
        for swap_workspace in (False, True):
            with self.subTest(workspace=swap_workspace):
                root = self.task("outer_workspace" if swap_workspace else "outer_nested")
                target = root / "workspace"
                if not swap_workspace:
                    target = target / "data"
                    target.mkdir()
                (target / "public.txt").write_bytes(b"public")
                expected = root / "expected"
                expected.mkdir()
                (expected / "secret.txt").write_bytes(b"SECRET_EXPECTED")
                original_child = Directory.child
                original_file = Directory.file
                swapped, opened = [], []

                def intercept(directory, name, **kwargs):
                    if directory.path / name == target and not swapped:
                        target.rename(root / "moved_original")
                        expected.rename(target)
                        swapped.append(True)
                    return original_child(directory, name, **kwargs)

                def track(directory, name, **kwargs):
                    opened.append(name)
                    return original_file(directory, name, **kwargs)

                with (
                    patch.object(Directory, "child", new=intercept),
                    patch.object(Directory, "file", new=track),
                ):
                    await self.failure(root)
                self.assertTrue(swapped)
                self.assertNotIn("secret.txt", opened)

    async def test_task_root_swap_after_discovery_is_rejected_before_manifest_read(self):
        root = self.task("task")
        expected = root / "expected"
        expected.mkdir()
        (expected / "task.json").write_text(
            json.dumps({"task_id": "fake", "instruction": "fake", "reward": 1}),
            encoding="utf-8",
        )
        (expected / "secret.txt").write_bytes(b"SECRET_EXPECTED")
        original = OrganizerTaskAdapter._load_checked
        original_file = Directory.file
        opened = []

        async def swap(adapter, path, expected_stat):
            moved = self.root / "moved"
            root.rename(moved)
            (moved / "expected").rename(root)
            return await original(adapter, path, expected_stat)

        def track(directory, name, **kwargs):
            opened.append(name)
            return original_file(directory, name, **kwargs)

        with (
            patch.object(OrganizerTaskAdapter, "_load_checked", new=swap),
            patch.object(Directory, "file", new=track),
        ):
            report = await DirectoryTaskLoader().load(self.root)
        self.assertEqual(report.tasks, ())
        self.assertEqual([issue.task_root for issue in report.issues], [root])
        self.assertEqual(opened, [])

    async def test_adapter_work_root_traversal_and_outside_are_isolated(self):
        root = self.task("task")
        sibling = self.task("sibling")
        for unsafe in (root / ".." / "sibling", sibling, root / "expected"):

            class Adapter:
                async def load_task(self, path):
                    return TaskSpec(
                        task_id=path.name,
                        instruction="x",
                        reward=1,
                        task_root=path,
                        work_root=unsafe if path == root else path,
                        artifacts=(),
                    )

            report = await DirectoryTaskLoader(Adapter()).load(self.root)
            self.assertEqual([task.task_id for task in report.tasks], ["sibling"])
            self.assertEqual([issue.task_root for issue in report.issues], [root])

    async def test_adapter_work_root_symlink_is_rejected(self):
        root = self.task("task")
        sibling = self.task("sibling")
        self.symlink(root / "redirect", sibling, directory=True)

        class Adapter:
            async def load_task(self, path):
                return TaskSpec(
                    task_id=path.name,
                    instruction="x",
                    reward=1,
                    task_root=path,
                    work_root=(path / "redirect") if path == root else path,
                    artifacts=(),
                )

        report = await DirectoryTaskLoader(Adapter()).load(self.root)
        self.assertEqual([task.task_id for task in report.tasks], ["sibling"])
        self.assertEqual([issue.task_root for issue in report.issues], [root])

    async def test_instruction_distinguishes_source_and_normalizes_test_paths(self):
        root = self.task(
            instruction="Run python3 main.py, then fix main.py; run python3 ./verify_custom.py to test; pytest ./checks.py"
        )
        for name in ("main.py", "verify_custom.py", "checks.py"):
            (root / "workspace" / name).write_bytes(b"pass")
        task = await OrganizerTaskAdapter().load_task(root)
        protected = {str(artifact.path): artifact.protected for artifact in task.artifacts}
        self.assertEqual(protected, {"main.py": False, "verify_custom.py": True, "checks.py": True})

    async def test_hard_links_are_rejected(self):
        root = self.task()
        source = self.root / "outside"
        source.write_bytes(b"other data")
        try:
            os.link(source, root / "workspace" / "linked")
        except OSError:
            self.skipTest("hard links unavailable")
        await self.failure(root)

    @unittest.skipUnless(os.name == "posix", "POSIX FIFO")
    async def test_special_file_is_rejected_without_blocking(self):
        root = self.task()
        os.mkfifo(root / "workspace" / "pipe")
        await asyncio.wait_for(self.failure(root), timeout=1)

    @unittest.skipUnless(os.name == "posix", "POSIX descriptor-relative traversal")
    async def test_file_swap_to_symlink_before_open_never_reads_target(self):
        root = self.task()
        path = root / "workspace" / "public.txt"
        path.write_bytes(b"public")
        outside = self.root / "secret"
        outside.write_bytes(b"secret")
        original = os.open

        def swap(name, flags, *args, **kwargs):
            if name == "public.txt":
                path.unlink()
                path.symlink_to(outside)
            return original(name, flags, *args, **kwargs)

        with patch("nsu_budget_agent.task_io._files.os.open", side_effect=swap):
            await self.failure(root)

    async def test_thread_io_and_task_cancellation_propagate(self):
        root = self.task()
        thread_ids = []
        original = Directory.file

        def capture(directory, name, **kwargs):
            thread_ids.append(threading.get_ident())
            return original(directory, name, **kwargs)

        with patch.object(Directory, "file", new=capture):
            await OrganizerTaskAdapter().load_task(root)
        self.assertTrue(thread_ids)
        self.assertNotIn(threading.get_ident(), thread_ids)
        # Fake adapter is an external async port; cancellation must not become LoadIssue.
        started = asyncio.Event()

        class WaitingAdapter:
            async def load_task(self, task_root):
                started.set()
                await asyncio.Event().wait()

        load = asyncio.create_task(DirectoryTaskLoader(WaitingAdapter()).load(self.root))
        await asyncio.wait_for(started.wait(), timeout=2)
        load.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await load
        self.assertTrue((root / "task.json").is_file())


if __name__ == "__main__":
    unittest.main()
