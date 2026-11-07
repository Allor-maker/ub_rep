"""Real A5/SDK + production B. Reference calculations use only public inputs."""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from tests.public_reference import public_reference

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import AttemptLimits, RunConfig, ToolLimits
from nsu_budget_agent.orchestration.runtime import AgentRunner
from nsu_budget_agent.solvers.composition import build
from nsu_budget_agent.task_io import DirectoryTaskLoader
from nsu_budget_agent.task_io._files import Directory, excluded
from nsu_budget_agent.task_io.workspace import artifact_file, work_directory


class ProductionPipelineTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from nsu_budget_agent.orchestration.sdk_probe import _offline_audit

        sys.addaudithook(_offline_audit)
        from openjiuwen.core.workflow import Workflow

        cls.sdk = Workflow

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.tasks = self.root / "tasks"
        self.tasks.mkdir()
        self.staging = self.root / "staging"
        self.staging.mkdir()

    def runner(self):
        now = asyncio.get_running_loop().time()
        config = RunConfig(
            total_budget_units=0,
            deadline_monotonic=now + 300,
            attempt_limits=AttemptLimits(
                max_cost_units=0,
                max_llm_calls=0,
                max_steps=3,
                max_duration_s=10,
                max_attempts_per_task=1,
                allowed_models=(),
                tool_limits=ToolLimits(
                    timeout_s=3,
                    max_read_bytes=1_048_576,
                    max_output_bytes=1_048_576,
                    max_write_bytes=1024,
                    max_process_memory_bytes=134_217_728,
                    max_process_count=1,
                ),
            ),
        )
        ledger = BudgetLedger(limit_units=0, deadline_monotonic=config.deadline_monotonic)
        runner = AgentRunner(
            config=config,
            ledger=ledger,
            components=build(ledger, config),
            staging_parent=self.staging,
        )
        self.addAsyncCleanup(runner.aclose)
        return runner

    def add_task(self, name, instruction, files=None, kind=None, check="exact_int", workspace=True):
        root = self.tasks / name
        root.mkdir()
        (root / "task.json").write_text(
            json.dumps(
                {
                    "task_id": name,
                    "instruction": instruction,
                    "reward": 1,
                    "type": kind,
                    "check": check,
                }
            ),
            encoding="utf-8",
        )
        work = root / "workspace" if workspace else root
        work.mkdir(exist_ok=True)
        for path, content in (files or {}).items():
            target = work.joinpath(*path.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content.encode("utf-8"))
        return root

    def synthetic_tasks(self):
        self.add_task("arithmetic", "Compute (17-5)*3.", kind="math", workspace=False)
        self.add_task(
            "table",
            "In account.csv, what is the sum of amount where label equals 'x'?",
            {"account.csv": "amount,label\n11,x\n99,y\n5,x\n"},
            kind="data_analysis",
        )
        self.add_task(
            "lookup",
            "In which city is the headquarters of Anne Example's employer?",
            {
                "archive/one.md": "# Anne Example\nEmployer: Wider Company\nBorn: 1992\n",
                "archive/two.md": "# Wider Company\nHeadquarters: Novel Town\n",
            },
            kind="search",
            check="exact_str",
        )
        self.add_task(
            "missing",
            "In which year was Anne Example born?",
            {"one.md": "# Anne Example\nEmployer: Wider Company\n"},
            kind="search",
            check="exact_str",
        )
        self.add_task(
            "unknown",
            "A warehouse starts with 10 boxes. It loses some boxes. How many boxes are left?",
            kind="math",
        )
        self.add_task(
            "coding",
            "Fix the available program.",
            {"main.py": "raise ValueError()"},
            kind="coding",
            check="unittest",
        )

    async def test_production_pipeline_solves_only_proven_synthetic_tasks(self):
        self.synthetic_tasks()
        # Reference markers must not even be opened during loading/staging/validation.
        for root in self.tasks.iterdir():
            work = root / "workspace" if (root / "workspace").is_dir() else root
            (work / "gold.md").write_text("forbidden reference", encoding="utf-8")
            (work / "expected.txt").write_text("forbidden reference", encoding="utf-8")
        original = Directory.file

        def guarded(directory, name, **kwargs):
            if name in ("gold.md", "expected.txt"):
                raise AssertionError("reference file was opened")
            return original(directory, name, **kwargs)

        with patch.object(Directory, "file", guarded):
            report = await self.runner().run(self.tasks)
        answers = {p.parent.name: p.read_bytes() for p in self.tasks.glob("*/answer.txt")}
        self.assertEqual(answers, {"arithmetic": b"36", "table": b"16", "lookup": b"Novel Town"})
        self.assertEqual(report.budget.spent_units, 0)
        self.assertEqual(report.budget.calls_used, 0)
        self.assertEqual(report.budget.held_units, 0)
        self.assertEqual(report.workflow_invocations, 6)
        self.assertEqual(list(self.staging.iterdir()), [])

    async def test_reference_variants_excluded_before_open_at_every_level(self):
        root = self.add_task(
            "public-task",
            "In which year was Demo Person born?",
            {
                "kb/profile.md": "# Demo Person\nBorn: 1975\n",
                "gold-standard.md": "# Demo Person\nBorn: 1999\n",
                "expected-answer.txt": "1999",
                "kb/gold-references/secret.md": "# Demo Person\nBorn: 1999\n",
                "nested/Expected-final/secret.txt": "1999",
                "nested/Gold—answers/secret.txt": "1999",
                "nested/expected2/secret.txt": "1999",
                "kb/golden.md": "# Other Person\nBorn: 1980\n",
            },
            kind="search",
            check="exact_str",
        )
        # A reference-named directory at discovery level must not become a task.
        hidden = self.tasks / "gold-hidden"
        hidden.mkdir()
        (hidden / "task.json").write_text("reference, not a task", encoding="utf-8")
        original_file, original_child = Directory.file, Directory.child

        def guarded_file(directory, name, **kwargs):
            if name != "task.json" and excluded(name):
                raise AssertionError("reserved reference file was opened")
            return original_file(directory, name, **kwargs)

        def guarded_child(directory, name, **kwargs):
            if excluded(name):
                raise AssertionError("reserved reference directory was entered")
            return original_child(directory, name, **kwargs)

        with (
            patch.object(Directory, "file", guarded_file),
            patch.object(Directory, "child", guarded_child),
        ):
            loaded = await DirectoryTaskLoader().load(self.tasks)
            self.assertEqual(len(loaded.tasks), 1)
            self.assertEqual(
                {a.path.as_posix() for a in loaded.tasks[0].artifacts},
                {"kb/profile.md", "kb/golden.md"},
            )
            report = await self.runner().run(self.tasks)
        self.assertEqual((root / "answer.txt").read_bytes(), b"1975")
        self.assertEqual(report.progress[0].status, "submitted")
        self.assertEqual(
            report.attempts[0].candidate.evidence_paths, (PurePosixPath("kb/profile.md"),)
        )
        self.assertEqual(report.budget.calls_used, 0)

    async def test_orders_and_missing_filter_through_real_sdk(self):
        duplicate = self.add_task(
            "duplicates",
            "The file x.csv has columns order_id,product,region (3 rows). "
            "How many orders of product 'gadget' came from region 'west'?",
            {"x.csv": "order_id,product,region\n42,gadget,west\n42,gadget,west\n43,gadget,west\n"},
            "data_analysis",
        )
        incomplete = self.add_task(
            "missing",
            "The file x.csv has columns order_id,product,region (1 rows). "
            "How many orders of product 'gadget' came from region 'west'?",
            {"x.csv": "order_id,product,region\n1,gadget,\n"},
            "data_analysis",
        )
        rows = self.add_task(
            "rows",
            "In x.csv, how many rows where product equals 'gadget'?",
            {"x.csv": "order_id,product\n42,gadget\n42,gadget\n43,gadget\n"},
            "data_analysis",
        )
        report = await self.runner().run(self.tasks)
        # Independent handwritten oracles: unique IDs {42,43}=2; three records=3.
        self.assertEqual((duplicate / "answer.txt").read_bytes(), b"2")
        self.assertEqual((rows / "answer.txt").read_bytes(), b"3")
        self.assertFalse((incomplete / "answer.txt").exists())
        self.assertEqual(sum(p.status == "submitted" for p in report.progress), 2)
        self.assertEqual(report.budget.calls_used, 0)

    async def test_cli_default_is_production_without_keys_or_fake(self):
        self.synthetic_tasks()
        env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "LLM_BASE_URL")}
        env.update(BUDGET="100000", PYTHONDONTWRITEBYTECODE="1", PYTHONPATH="src")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "run_agent.py",
            "--tasks-dir",
            str(self.tasks),
            "--budget-unit",
            "units_x10",
            "--staging-dir",
            str(self.staging),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 90)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        self.assertEqual(process.returncode, 0, stderr.decode(errors="replace")[-2000:])
        summary = next(
            json.loads(line)
            for line in stdout.decode().splitlines()
            if line.startswith('{"tasks":')
        )
        self.assertEqual(summary["saved"], 3)
        self.assertEqual(summary["limit_units"], 100_000)
        self.assertEqual(summary["spent_units"], 0)
        self.assertEqual(summary["accounting_mode"], "strict")

    async def test_competition_cli_defaults_to_competition_tokens(self):
        root = self.add_task("arithmetic", "Compute (17-5)*3.", kind="math", workspace=False)
        env = {k: v for k, v in os.environ.items() if k not in ("LLM_API_KEY", "LLM_BASE_URL")}
        env.update(
            BUDGET="100000",
            PYTHONDONTWRITEBYTECODE="1",
            PYTHONPATH="src",
            TMPDIR=str(self.staging),
            TEMP=str(self.staging),
            TMP=str(self.staging),
        )
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(Path("run_agent.py").absolute()),
            "--tasks-dir",
            str(self.tasks),
            cwd=self.staging,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), 90)
        except BaseException:
            process.kill()
            await process.wait()
            raise
        self.assertEqual(process.returncode, 0, stderr.decode(errors="replace")[-2000:])
        summary = next(
            json.loads(line)
            for line in stdout.decode().splitlines()
            if line.startswith('{"tasks":')
        )
        self.assertEqual(summary["limit_units"], 1_000_000)
        self.assertEqual(summary["spent_units"], 0)
        self.assertEqual(summary["held_units"], 0)
        self.assertEqual(summary["accounting_mode"], "strict")
        self.assertEqual(summary["saved"], 1)
        self.assertEqual((root / "answer.txt").read_bytes(), b"36")

    async def public_dataset(self, relative, solved):
        # Local datasets are optional inputs, never copied wholesale or modified.
        source = Path.cwd().parent / relative
        if not source.is_dir():
            self.skipTest(
                "public local dataset not supplied; portable production fixtures still run"
            )
        report = await DirectoryTaskLoader().load(source)
        self.assertEqual(report.issues, ())
        references = {}
        for task in report.tasks:
            files = {}
            with work_directory(task) as directory:
                for artifact in task.artifacts:
                    # No expected/gold reads, no copying source manifests or old outputs.
                    with artifact_file(directory, artifact.path) as descriptor:
                        chunks = []
                        while chunk := os.read(descriptor, 65_536):
                            chunks.append(chunk)
                        files[artifact.path.as_posix()] = b"".join(chunks).decode("utf-8-sig")
            self.add_task(task.task_id, task.instruction, files, task.type_hint, task.check_hint)
            references[task.task_id] = public_reference(task.instruction, files)
        run = await self.runner().run(self.tasks)
        correct = 0
        for task_id, answer in references.items():
            path = self.tasks / task_id / "answer.txt"
            if answer is None:
                self.assertFalse(path.exists(), task_id)
            else:
                self.assertTrue(path.exists(), task_id)
                self.assertEqual(path.read_bytes(), answer.encode("utf-8"), task_id)
                correct += 1
        self.assertEqual(correct, solved)
        self.assertEqual(sum(r.status == "saved" for r in run.receipts), solved)
        self.assertEqual(run.budget.calls_used, 0)
        print(f"PUBLIC DATASET {relative}: verified={correct}/{len(report.tasks)}, LLM calls=0")

    async def test_original_public_examples(self):
        await self.public_dataset("task-dir/task-dir", 6)

    async def test_local_synthetic_smoke(self):
        await self.public_dataset("nsu_benchmark_smoke_12_tasks/tasks", 6)

    async def test_local_synthetic_benchmark(self):
        await self.public_dataset("nsu_synthetic_benchmark/data/tasks", 60)
