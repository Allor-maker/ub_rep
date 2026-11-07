"""C02/C11 contract flow stops at PlanDecision, with a fake B solver."""

import json
import tempfile
import unittest
from pathlib import Path

from nsu_budget_agent.contracts.models import (
    AttemptLimits,
    BudgetSnapshot,
    RunConfig,
    SolverOffer,
    TaskProgress,
    ToolLimits,
)
from nsu_budget_agent.contracts.ports import Planner, SolverRegistry, TaskLoader
from nsu_budget_agent.planning import BaselinePlanner, StaticSolverRegistry
from nsu_budget_agent.task_io import DirectoryTaskLoader


class GenericDouble:
    solver_id = "generic"

    def assess(self, task, tools):
        return SolverOffer(
            task_id=task.task_id,
            solver_id=self.solver_id,
            applicability="unknown",
            required_tools=(),
        )

    async def step(self, ctx, history):
        raise AssertionError("execution belongs to A5")


class A4ContractTests(unittest.IsolatedAsyncioTestCase):
    async def test_unknown_type_without_workspace_and_coverage_after_submission(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            for name, reward in (("rich", 1000), ("poor", 0)):
                directory = root / name
                directory.mkdir()
                (directory / "task.json").write_text(
                    json.dumps(
                        {
                            "task_id": name,
                            "instruction": "Read renamed-input.txt",
                            "task_type": "future_kind",
                            "reward": reward,
                            "expected": "not an agent input",
                        }
                    ),
                    encoding="utf-8",
                )
                (directory / "renamed-input.txt").write_bytes(b"public")
            loader: TaskLoader = DirectoryTaskLoader()
            registry: SolverRegistry = StaticSolverRegistry()
            registry.register(GenericDouble())
            report = await loader.load(root)
            self.assertEqual(report.issues, ())
            self.assertEqual(len(report.tasks), 2)
            self.assertTrue(all(task.work_root == task.task_root for task in report.tasks))
            offers = tuple(
                solver.assess(task, ())
                for task in report.tasks
                for solver in registry.all_solvers()
            )
            config = RunConfig(
                total_budget_units=1000,
                deadline_monotonic=100,
                attempt_limits=AttemptLimits(
                    max_cost_units=400,
                    max_llm_calls=1,
                    max_steps=3,
                    max_duration_s=20,
                    allowed_models=("qwen3.8-27b",),
                    tool_limits=ToolLimits(
                        timeout_s=1,
                        max_read_bytes=100,
                        max_output_bytes=100,
                        max_write_bytes=100,
                        max_process_memory_bytes=1000,
                        max_process_count=1,
                    ),
                ),
            )
            snapshot = BudgetSnapshot(
                limit_units=1000, spent_units=0, held_units=0, calls_used=0, blocked=False
            )
            planner: Planner = BaselinePlanner()
            first = planner.choose(report.tasks, offers, (), snapshot, config, 10)
            self.assertEqual(first.task_id, "rich")
            progress = (
                TaskProgress(task_id="rich", status="submitted", attempted_solvers=("generic",)),
            )
            second = planner.choose(report.tasks, offers, progress, snapshot, config, 10)
            self.assertEqual(second.task_id, "poor")
            complete = (*progress, TaskProgress(task_id="poor", status="submitted"))
            self.assertIsNone(planner.choose(report.tasks, offers, complete, snapshot, config, 10))
            self.assertEqual(list(root.rglob("answer.txt")), [])
