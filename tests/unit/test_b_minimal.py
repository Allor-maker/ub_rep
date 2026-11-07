"""Independent arithmetic oracles and adversarial local evidence; no model/gold."""

import dataclasses
import hashlib
import os
import tempfile
import time
import unittest
from decimal import localcontext
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from nsu_budget_agent.contracts.models import (
    ArtifactRef,
    AttemptBudget,
    AttemptResult,
    CandidateAnswer,
    StepRecord,
    TaskSpec,
    ToolCall,
    ToolContext,
    ToolLimits,
    ToolResult,
)
from nsu_budget_agent.contracts.ports import SolverContext
from nsu_budget_agent.solvers.deterministic import DeterministicSolver
from nsu_budget_agent.solvers.plans import plan
from nsu_budget_agent.submission import V1AcceptancePolicy
from nsu_budget_agent.task_io.workspace import manifest, snapshot
from nsu_budget_agent.tools import LocalToolExecutor
from nsu_budget_agent.validation import LocalValidator

LIMITS = ToolLimits(
    timeout_s=2,
    max_read_bytes=100_000,
    max_output_bytes=100_000,
    max_write_bytes=1000,
    max_process_memory_bytes=10_000_000,
    max_process_count=1,
)


class LocalTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.tools = LocalToolExecutor()

    def task(self, instruction="Compute 2 + 3.", files=None, **changes):
        artifacts = []
        for name, content in (files or {}).items():
            path = self.root.joinpath(*PurePosixPath(name).parts)
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = content if isinstance(content, bytes) else content.encode("utf-8")
            path.write_bytes(raw)
            artifacts.append(
                ArtifactRef(
                    path=PurePosixPath(name),
                    size_bytes=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest(),
                    protected=name.startswith("tests/"),
                )
            )
        return TaskSpec(
            task_id="task",
            task_root=self.root,
            work_root=self.root,
            instruction=instruction,
            reward=1,
            artifacts=tuple(artifacts),
            **changes,
        )

    def context(self, task):
        return ToolContext(
            attempt_id="attempt",
            workspace_root=self.root,
            source_artifacts=task.artifacts,
            protected_paths=tuple(a.path for a in task.artifacts if a.protected),
            deadline_monotonic=time.monotonic() + 30,
            limits=LIMITS,
        )

    async def calculate(self, instruction, files=None, **fields):
        task = self.task(instruction, files, **fields)
        result = await self.tools.execute("call", plan(task).call, self.context(task))
        self.assertEqual(result.status, "ok", result.error)
        return result.data["answer"]

    async def test_exact_arithmetic_precedence_decimal_and_floor(self):
        for expression, expected in (
            ("2 + 3 * 7", "23"),
            ("(2+3)**3", "125"),
            ("0.1+0.2", "0.3"),
            ("-7//3", "-3"),
            ("7%3", "1"),
            ("1/8", "0.125"),
        ):
            self.assertEqual(await self.calculate(f"Compute {expression}."), expected)

    async def test_arithmetic_rejects_code_bombs_undefined_and_ambiguous_rounding(self):
        task = self.task()
        for expression in (
            "__import__('os').getcwd()",
            "2**999999",
            "(lambda: 1)()",
            "1/0",
            "1/3",
            "True+1",
            "'abc' * 100",
            "1e300",
        ):
            result = await self.tools.execute(
                "call",
                ToolCall(tool_name="local_math", arguments={"expression": expression}),
                self.context(task),
            )
            self.assertEqual(result.status, "error", expression)

    async def test_warehouse_families_and_unknown_operations(self):
        cases = (
            (
                "A warehouse starts with 99 boxes. It ships 7 boxes per day for 3 days. It receives 2 deliveries of 10 boxes each. How many boxes does it have at the end?",
                "98",
            ),
            (
                "A warehouse starts with 53 boxes, receives 11 new boxes, discards 7 damaged boxes, then sends 2 batches of 9 boxes. What is the final stock?",
                "39",
            ),
            (
                "A warehouse starts with 53 boxes. It is split equally into 3 warehouses and keeps one share (rounded down). The stock is doubled, then split into 4 equal shares, keeping one share (rounded down). How many boxes are left?",
                "8",
            ),
            (
                "A warehouse starts with 53 boxes. It ships 2 boxes per day for 3 days. The remainder is divided equally among 4 sites, and this warehouse keeps one share (rounded down), then receives an extra 3 boxes. What is its final stock?",
                "14",
            ),
        )
        for instruction, expected in cases:
            self.assertEqual(await self.calculate(instruction), expected)
        for instruction in (
            "A warehouse starts with 53 boxes. It loses some boxes. How many boxes are left?",
            "Compute 2+3. Also double the result.",
        ):
            with self.assertRaises(ValueError):
                plan(self.task(instruction))

    async def test_csv_reordered_columns_quoted_values_and_empty_sum(self):
        files = {
            "reports/new-name.csv": 'month,units,product,price,region,order_id\n2,3,"red, large",2.5,west,1\n3,9,blue,1.2,east,2\n'
        }
        prefix = "The file reports/new-name.csv has columns order_id,region,product,month,units,price (2 rows). "
        self.assertEqual(
            await self.calculate(
                prefix
                + "How many units of product 'red, large' were sold in months 2 through 2 inclusive?",
                files,
            ),
            "3",
        )
        self.assertEqual(
            await self.calculate(
                prefix
                + "What is the total revenue (units multiplied by price) for product 'red, large' in months 1 through 3 inclusive?",
                files,
            ),
            "7.5",
        )
        self.assertEqual(
            await self.calculate(
                prefix + "How many orders of product 'missing' came from region 'east'?", files
            ),
            "0",
        )

    async def test_generic_csv_columns_and_decimal_context_independence(self):
        with localcontext() as context:
            context.prec, context.Emax, context.Emin = 1, 1, -1
            self.assertEqual(
                await self.calculate(
                    "In ledger.csv, what is the sum of amount where label equals 'chosen'?",
                    {"ledger.csv": "label,amount\nchosen,100.25\nother,3\nchosen,0.5\n"},
                ),
                "100.75",
            )

    async def test_csv_bad_shapes_numbers_and_incomplete_data_refused(self):
        instruction = "The file x.csv has columns product,month,units (1 rows). How many units of product 'a' were sold in months 1 through 2 inclusive?"
        for text in (
            "product,month,units\na,1\n",
            "product,month,units\na,1,NaN\n",
            "product,month,units\na,1,4\na,2,3\n",
            "product,month,units,units\na,1,4,5\n",
            'product,month,units\n"a,1,2\n',
        ):
            task = self.task(instruction, {"x.csv": text})
            result = await self.tools.execute("call", plan(task).call, self.context(task))
            self.assertEqual(result.status, "error")

    async def test_order_count_is_distinct_and_rows_are_separate(self):
        data = "order_id,product,region\n42,gadget,west\n42,gadget,west\n43,gadget,west\n44,other,west\n"
        instruction = (
            "The file orders.csv has columns order_id,product,region (4 rows). "
            "How many orders of product 'gadget' came from region 'west'?"
        )
        task = self.task(instruction, {"orders.csv": data}, check_hint="exact_int")
        result = await self.tools.execute("call", plan(task).call, self.context(task))
        # Hand oracle: matching order IDs are {42, 43}; there are three matching records.
        self.assertEqual(result.data["answer"], "2")
        rows = ToolCall(
            tool_name="query_csv",
            arguments={
                "path": "orders.csv",
                "aggregate": "count_rows",
                "columns": [],
                "filters": [{"column": "product", "op": "eq", "value": "gadget"}],
            },
        )
        self.assertEqual(
            (await self.tools.execute("rows", rows, self.context(task))).data["answer"], "3"
        )
        alias = ToolCall(tool_name="query_csv", arguments={**rows.arguments, "aggregate": "count"})
        self.assertEqual(
            (await self.tools.execute("rows", alias, self.context(task))).data["answer"], "3"
        )
        self.assertEqual((await self.validate(task, "2", ["orders.csv"]))[1].status, "accept")
        self.assertEqual((await self.validate(task, "3", ["orders.csv"]))[1].status, "reject")

    async def test_explicit_count_rows_keeps_row_semantics_in_solver_and_validator(self):
        task = self.task(
            "In orders.csv, how many rows where product equals 'gadget'?",
            {"orders.csv": "order_id,product\n42,gadget\n42,gadget\n43,other\n"},
            check_hint="exact_int",
        )
        result = await self.tools.execute("call", plan(task).call, self.context(task))
        self.assertEqual(result.data["answer"], "2")
        validation, decision = await self.validate(task, "2", ["orders.csv"])
        self.assertEqual(decision.status, "accept")
        self.assertEqual(
            next(c.status for c in validation.checks if c.check_id == "csv_independent"), "passed"
        )
        self.assertEqual((await self.validate(task, "1", ["orders.csv"]))[1].status, "reject")

    async def test_orders_require_declared_identity_and_reject_ambiguous_ids(self):
        no_id = self.task(
            "The file x.csv has columns product,region (1 rows). "
            "How many orders of product 'gadget' came from region 'west'?",
            {"x.csv": "product,region\ngadget,west\n"},
        )
        with self.assertRaises(ValueError):
            plan(no_id)
        for data in (
            "order_id,product,region\n,gadget,west\n2,gadget,west\n",
            "order_id,product,region\n 42,gadget,west\n42,gadget,west\n",
            "order_id,product,region\nX,gadget,west\nx,gadget,west\n",
            "order_id,product,region\n42,gadget,west\n42,gadget,east\n",
        ):
            task = self.task(
                "The file x.csv has columns order_id,product,region (2 rows). "
                "How many orders of product 'gadget' came from region 'west'?",
                {"x.csv": data},
            )
            result = await self.tools.execute("call", plan(task).call, self.context(task))
            self.assertEqual(result.status, "error", data)

    async def test_empty_filter_values_fail_closed_for_all_rows_and_aggregates(self):
        cases = (
            (
                "The file x.csv has columns order_id,product,region (1 rows). "
                "How many orders of product 'gadget' came from region 'west'?",
                "order_id,product,region\n1,gadget,\n",
                "0",
            ),
            (
                "The file x.csv has columns order_id,product,region (1 rows). "
                "How many orders of product 'gadget' came from region 'west'?",
                "order_id,product,region\n1,,west\n",
                "0",
            ),
            (
                "The file x.csv has columns order_id,product,region (1 rows). "
                "How many orders of product 'gadget' came from region 'west'?",
                "order_id,product,region\n1,other, \n",
                "0",
            ),
            (
                "In x.csv, what is the sum of amount where label equals 'x'?",
                "label,amount\n,4\n",
                "0",
            ),
            ("In x.csv, how many rows where label equals 'x'?", "label,amount\n  ,4\n", "0"),
            (
                "The file x.csv has columns product,month,units (1 rows). "
                "How many units of product 'a' were sold in months 1 through 2 inclusive?",
                "product,month,units\na,,4\n",
                "0",
            ),
        )
        for instruction, data, candidate in cases:
            with self.subTest(data=data):
                task = self.task(instruction, {"x.csv": data}, check_hint="exact_int")
                reply = await self.tools.execute("call", plan(task).call, self.context(task))
                self.assertEqual(reply.status, "error")
                validation, decision = await self.validate(task, candidate, ["x.csv"])
                self.assertEqual(decision.status, "reject")
                self.assertEqual(
                    next(c.status for c in validation.checks if c.check_id == "csv_independent"),
                    "failed",
                )

    async def test_independent_validator_catches_old_shared_count_bug(self):
        instruction = (
            "The file x.csv has columns order_id,product,region (2 rows). "
            "How many orders of product 'gadget' came from region 'west'?"
        )
        task = self.task(
            instruction,
            {"x.csv": "order_id,product,region\n42,gadget,west\n42,gadget,west\n"},
            check_hint="exact_int",
        )
        original = self.tools.execute

        async def flawed_aggregation(call_id, call, ctx):
            result = await original(call_id, call, ctx)
            if call.tool_name == "query_csv" and result.status == "ok":
                # Inject the former bug: both matching records are counted as orders.
                return dataclasses.replace(
                    result, data={"answer": "2", "evidence_paths": ["x.csv"]}
                )
            return result

        with patch.object(self.tools, "execute", side_effect=flawed_aggregation):
            validation, decision = await self.validate(task, "2", ["x.csv"])
        checks = {c.check_id: c.status for c in validation.checks}
        self.assertEqual(checks["semantics"], "passed")
        self.assertEqual(checks["csv_independent"], "failed")
        self.assertEqual(decision.status, "reject")
        self.assertEqual((await self.validate(task, "1", ["x.csv"]))[1].status, "accept")

    async def test_independent_validator_catches_forged_zero_for_missing_filter(self):
        task = self.task(
            "The file x.csv has columns order_id,product,region (1 rows). "
            "How many orders of product 'gadget' came from region 'west'?",
            {"x.csv": "order_id,product,region\n1,gadget,\n"},
            check_hint="exact_int",
        )
        original = self.tools.execute

        async def flawed_aggregation(call_id, call, ctx):
            if call.tool_name == "query_csv":
                return ToolResult(
                    call_id=call_id,
                    tool_name="query_csv",
                    status="ok",
                    data={"answer": "0", "evidence_paths": ["x.csv"]},
                )
            return await original(call_id, call, ctx)

        with patch.object(self.tools, "execute", side_effect=flawed_aggregation):
            validation, decision = await self.validate(task, "0", ["x.csv"])
        checks = {c.check_id: c.status for c in validation.checks}
        self.assertEqual(checks["semantics"], "passed")
        self.assertEqual(checks["csv_independent"], "failed")
        self.assertEqual(decision.status, "reject")

    async def test_validator_rejects_mismatched_tool_response_identity(self):
        task = self.task(
            "The file x.csv has columns order_id,product,region (1 rows). "
            "How many orders of product 'gadget' came from region 'west'?",
            {"x.csv": "order_id,product,region\n42,gadget,west\n"},
            check_hint="exact_int",
        )
        original = self.tools.execute

        async def wrong_identity(call_id, call, ctx):
            result = await original(call_id, call, ctx)
            return (
                dataclasses.replace(result, call_id="other")
                if call.tool_name == "query_csv"
                else result
            )

        with patch.object(self.tools, "execute", side_effect=wrong_identity):
            validation, decision = await self.validate(task, "1", ["x.csv"])
        self.assertEqual(
            next(c.status for c in validation.checks if c.check_id == "semantics"), "inconclusive"
        )
        self.assertEqual(decision.status, "reject")

    async def test_csv_audit_does_not_require_returning_full_csv_as_tool_output(self):
        data = "order_id,product,region\n" + "42,gadget,west\n" * 100
        instruction = (
            "The file x.csv has columns order_id,product,region (100 rows). "
            "How many orders of product 'gadget' came from region 'west'?"
        )
        task = self.task(instruction, {"x.csv": data}, check_hint="exact_int")
        ctx = dataclasses.replace(
            self.context(task), limits=dataclasses.replace(LIMITS, max_output_bytes=100)
        )
        candidate = CandidateAnswer(text="1", evidence_paths=(PurePosixPath("x.csv"),))
        artifacts = manifest(self.root, frozenset(), max_bytes=100_000)
        result = AttemptResult(
            task_id=task.task_id,
            attempt_id=ctx.attempt_id,
            solver_id="local_data_analysis",
            status="candidate",
            workspace_root=self.root,
            candidate=candidate,
            snapshot_sha256=snapshot(candidate, artifacts),
        )
        self.assertGreater(len(data.encode("utf-8")), ctx.limits.max_output_bytes)
        validation = await LocalValidator(self.tools).validate(task, result, ctx)
        self.assertEqual(V1AcceptancePolicy().decide(result, validation).status, "accept")
        self.assertEqual(
            next(c.status for c in validation.checks if c.check_id == "csv_independent"), "passed"
        )

    async def test_markdown_relations_entity_identity_and_directory_scope(self):
        files = {
            "profiles/a.md": "# Ada Example\nBorn: 1987\nEmployer: Novel Co\nMentor: Bob Example\n",
            "profiles/b.md": "# Novel Co\nHeadquarters: New City\nFounded: 1973\n",
            "profiles/c.md": "# Bob Example\nBorn: 1962\n",
            "other/decoy.md": "# Ada Example\nBorn: 2000\n",
        }
        prefix = "The directory profiles/ contains a small knowledge base of Markdown files. "
        for question, expected in (
            ("In which year was Ada Example born?", "1987"),
            ("In which city is the headquarters of Ada Example's employer?", "New City"),
            ("In which year was Ada Example's employer founded?", "1973"),
            ("In which year was Ada Example's mentor born?", "1962"),
        ):
            self.assertEqual(await self.calculate(prefix + question, files), expected)

    async def test_missing_or_conflicting_markdown_facts_refused(self):
        for files in (
            {"a.md": "# Person\nEmployer: Missing\n"},
            {"a.md": "# Person\nBorn: 1990\n", "b.md": "# Person\nBorn: 1991\n"},
            {"a.md": "# Person\nBorn: 1990\nBorn: 1991\n"},
        ):
            task = self.task("In which year was Person born?", files)
            result = await self.tools.execute("call", plan(task).call, self.context(task))
            self.assertEqual(result.status, "error")
            for name in files:
                (self.root / name).unlink()

    async def test_paths_reserved_files_hashes_and_unknown_tools(self):
        task = self.task(
            files={"public.txt": "safe", "expected.txt": "secret", "gold.md": "secret"}
        )
        ctx = self.context(task)
        for path in (
            "../public.txt",
            "/tmp/public.txt",
            "expected.txt",
            "gold.md",
            "task.json",
            "answer.txt",
            "unknown.txt",
            "a/../public.txt",
        ):
            result = await self.tools.execute(
                "call", ToolCall(tool_name="read_file", arguments={"path": path}), ctx
            )
            self.assertEqual(result.status, "error", path)
        result = await self.tools.execute(
            "call", ToolCall(tool_name="read_file", arguments={"path": "public.txt"}), ctx
        )
        self.assertEqual(result.data["text"], "safe")
        (self.root / "public.txt").write_text("changed", encoding="utf-8")
        result = await self.tools.execute(
            "call", ToolCall(tool_name="read_file", arguments={"path": "public.txt"}), ctx
        )
        self.assertEqual(result.error.code, "denied")
        result = await self.tools.execute(
            "call", ToolCall(tool_name="shell", arguments={"command": "ls"}), ctx
        )
        self.assertEqual(result.error.code, "unsupported")
        self.assertTrue(all(not t.mutates_files for t in self.tools.catalog()))

    async def test_reference_aliases_denied_at_every_tool_path_component(self):
        files = {
            "gold-standard.md": "reference",
            "expected-answer.txt": "reference",
            "folder/Gold—answer/data.txt": "reference",
            "folder/EXPECTED2/data.txt": "reference",
            "folder/gold answer/data.txt": "reference",
            "folder/expected_answer/data.txt": "reference",
            "golden.txt": "ordinary public input",
        }
        task = self.task(files=files)
        for path in files:
            reply = await self.tools.execute(
                "call",
                ToolCall(tool_name="read_file", arguments={"path": path}),
                self.context(task),
            )
            if path == "golden.txt":
                self.assertEqual(reply.data["text"], "ordinary public input")
            else:
                self.assertEqual(reply.status, "error", path)

    async def test_read_output_and_time_limits(self):
        task = self.task(files={"large.txt": "x" * 1000})
        call = ToolCall(tool_name="read_file", arguments={"path": "large.txt"})
        for field, value in (("max_read_bytes", 10), ("max_output_bytes", 10)):
            ctx = dataclasses.replace(
                self.context(task), limits=dataclasses.replace(LIMITS, **{field: value})
            )
            result = await self.tools.execute("call", call, ctx)
            self.assertEqual(result.error.code, "resource_limit")
        result = await self.tools.execute(
            "call",
            call,
            dataclasses.replace(self.context(task), deadline_monotonic=time.monotonic() - 1),
        )
        self.assertEqual(result.error.code, "timeout")

    async def test_symlink_read_refused(self):
        task = self.task(files={"public.txt": "safe"})
        path = self.root / "public.txt"
        path.unlink()
        target = self.root / "outside.txt"
        target.write_text("secret", encoding="utf-8")
        try:
            path.symlink_to(target)
        except OSError as error:
            if os.name == "nt" and getattr(error, "winerror", None) == 1314:
                self.skipTest("Windows symlink privilege unavailable")
            raise
        result = await self.tools.execute(
            "call",
            ToolCall(tool_name="read_file", arguments={"path": "public.txt"}),
            self.context(task),
        )
        self.assertEqual(result.status, "error")

    async def validate(self, task, text, evidence=(), **changes):
        ctx = self.context(task)
        candidate = CandidateAnswer(
            text=text, evidence_paths=tuple(PurePosixPath(p) for p in evidence)
        )
        artifacts = manifest(self.root, frozenset(ctx.protected_paths), max_bytes=100000)
        result = AttemptResult(
            task_id="task",
            attempt_id="attempt",
            solver_id="local_math",
            status="candidate",
            workspace_root=self.root,
            candidate=candidate,
            snapshot_sha256=snapshot(candidate, artifacts),
            **changes,
        )
        validation = await LocalValidator(self.tools).validate(task, result, ctx)
        return validation, V1AcceptancePolicy().decide(result, validation)

    async def test_validator_rejects_wrong_nonempty_answer_unknown_hint_and_scope(self):
        for answer, hint, accepted in (
            ("5", "exact_int", True),
            ("6", "exact_int", False),
            ("5", "unknown-check", False),
            ("5\n", "exact_int", False),
        ):
            task = self.task(check_hint=hint)
            _, decision = await self.validate(task, answer)
            self.assertEqual(decision.status == "accept", accepted)
        _, decision = await self.validate(self.task("Predict tomorrow's weather."), "sunny")
        self.assertEqual(decision.status, "reject")

    async def test_validator_requires_semantic_evidence_and_protected_integrity(self):
        task = self.task(
            "In which year was Person born?",
            {"a.md": "# Person\nBorn: 1980\n", "tests/test_visible.py": "original"},
            check_hint="exact_str",
        )
        _, decision = await self.validate(task, "1980", ["a.md"])
        self.assertEqual(decision.status, "accept")
        _, decision = await self.validate(task, "1980")
        self.assertEqual(decision.status, "reject")
        (self.root / "tests/test_visible.py").write_text("changed", encoding="utf-8")
        validation, decision = await self.validate(task, "1980", ["a.md"])
        self.assertEqual(decision.status, "reject")
        self.assertEqual(
            next(c.status for c in validation.checks if c.check_id == "protected_files"), "failed"
        )

    async def test_stateless_history_and_failed_tool_no_candidate(self):
        task = self.task()
        solver = DeterministicSolver("math")
        offer = solver.assess(task, self.tools.catalog())
        self.assertEqual(offer.expected_cost_units, 0)
        self.assertIsNone(offer.success_probability)
        self.assertIsNone(offer.expected_duration_s)
        budget = AttemptBudget(
            max_cost_units=0,
            max_llm_calls=0,
            max_steps=4,
            deadline_monotonic=time.monotonic() + 30,
            allowed_models=(),
            tool_limits=LIMITS,
        )

        class NoLLM:
            async def complete(self, request):
                raise AssertionError("offline solver must not call LLM")

        ctx = SolverContext(
            task=task,
            attempt_id="attempt",
            step_index=0,
            budget=budget,
            tools=self.tools.catalog(),
            llm=NoLLM(),
        )
        action = await solver.step(ctx, ())
        self.assertEqual(action, await solver.step(ctx, ()))
        reply = await self.tools.execute("call", action, self.context(task))
        history = (StepRecord(step_index=0, call=action, result=reply),)
        self.assertEqual((await solver.step(ctx, history)).text, "5")
        truncated = dataclasses.replace(reply, truncated=True)
        refused = await solver.step(ctx, (dataclasses.replace(history[0], result=truncated),))
        self.assertEqual(refused.status, "blocked_by_evidence")
