"""C01: executable Contracts v1 boundary checks, using only the standard library."""

import ast
import asyncio
import dataclasses
import inspect
import json
import os
import subprocess
import sys
import types
import typing
import unittest
from pathlib import Path, PurePosixPath

from nsu_budget_agent.contracts import models as m
from nsu_budget_agent.contracts import ports as p

ROOT = Path(__file__).resolve().parents[2]
MODEL = "qwen3.8-27b"
SHA = "a" * 64


def examples():
    """One representative valid instance of every DTO, including explicit unknowns."""
    error = m.ErrorInfo(code="new_external_code", message="diagnostic")
    artifact = m.ArtifactRef(path=m.parse_rel_path("new-name.txt"), size_bytes=3, sha256=SHA)
    task = m.TaskSpec(
        task_id="task",
        instruction="Read the file",
        reward=0,
        task_root=ROOT,
        work_root=ROOT,
        artifacts=(artifact,),
        type_hint="unseen-format-v42",
    )
    limits = m.ToolLimits(
        timeout_s=1.0,
        max_read_bytes=1024,
        max_output_bytes=1024,
        max_write_bytes=1024,
        max_process_memory_bytes=1048576,
        max_process_count=1,
    )
    budget = m.AttemptBudget(
        max_cost_units=1000,
        max_llm_calls=2,
        max_steps=3,
        deadline_monotonic=100.0,
        allowed_models=(MODEL,),
        tool_limits=limits,
    )
    attempt_limits = m.AttemptLimits(
        max_cost_units=1000,
        max_llm_calls=2,
        max_steps=3,
        max_duration_s=10.0,
        allowed_models=(MODEL,),
        tool_limits=limits,
    )
    usage = m.UsageRecord(
        request_id="req",
        attempt_id="attempt",
        model=MODEL,
        status="exact",
        input_tokens=120,
        output_tokens=40,
        reasoning_tokens=20,
    )
    message = m.ChatMessage(role="user", content="Read")
    spec = m.ToolSpec(
        name="read",
        description="read text",
        input_schema={"type": "object"},
        output_schema={"type": "string"},
        mutates_files=False,
    )
    call = m.ToolCall(tool_name="read", arguments={"path": "new-name.txt"})
    result = m.ToolResult(call_id="call", tool_name="read", status="ok", data="abc")
    candidate = m.CandidateAnswer(text="  abc\n")
    attempt = m.AttemptResult(
        attempt_id="attempt",
        task_id="task",
        solver_id="generic",
        status="candidate",
        workspace_root=ROOT,
        candidate=candidate,
        snapshot_sha256=SHA,
        steps_used=2,
    )
    check = m.CheckResult(check_id="format", status="passed", blocking=True, message="ok")
    return {
        "ErrorInfo": error,
        "ArtifactRef": artifact,
        "TaskSpec": task,
        "SolverOffer": m.SolverOffer(
            task_id="task", solver_id="generic", applicability="unknown", required_tools=("read",)
        ),
        "ToolLimits": limits,
        "AttemptBudget": budget,
        "UsageRecord": usage,
        "BudgetLease": m.BudgetLease(
            request_id="req",
            attempt_id="attempt",
            model=MODEL,
            input_bound_tokens=120,
            output_cap_tokens=60,
            state="reserved",
        ),
        "BudgetSnapshot": m.BudgetSnapshot(
            limit_units=1000, spent_units=520, held_units=0, calls_used=1, blocked=False
        ),
        "ProviderProfile": m.ProviderProfile(),
        "ChatMessage": message,
        "LLMRequest": m.LLMRequest(model=MODEL, messages=(message,), max_output_tokens=60),
        "LLMReply": m.LLMReply(request_id="req", text="", finish="stop", usage=usage),
        "ToolSpec": spec,
        "ToolCall": call,
        "ToolContext": m.ToolContext(
            attempt_id="attempt",
            workspace_root=ROOT,
            source_artifacts=(artifact,),
            protected_paths=(),
            deadline_monotonic=100.0,
            limits=limits,
        ),
        "ToolResult": result,
        "CandidateAnswer": candidate,
        "StopAttempt": m.StopAttempt(status="no_progress", reason="No evidence"),
        "StepRecord": m.StepRecord(step_index=0, call=call, result=result),
        "FileChange": m.FileChange(path=m.parse_rel_path("new.py"), operation="write", sha256=SHA),
        "AttemptResult": attempt,
        "CheckResult": check,
        "ValidationResult": m.ValidationResult(
            attempt_id="attempt", snapshot_sha256=SHA, checks=(check,), status="passed"
        ),
        "AcceptanceDecision": m.AcceptanceDecision(
            attempt_id="attempt", snapshot_sha256=SHA, status="accept", reason="Checks passed"
        ),
        "PublicationPlan": m.PublicationPlan(answer_path=ROOT / "answer.txt"),
        "SubmissionReceipt": m.SubmissionReceipt(
            task_id="task",
            attempt_id="attempt",
            status="saved",
            answer_path=ROOT / "answer.txt",
            answer_sha256=SHA,
        ),
        "LoadIssue": m.LoadIssue(task_root=ROOT, error=error),
        "LoadReport": m.LoadReport(tasks=(task,), issues=()),
        "TaskProgress": m.TaskProgress(task_id="task", status="pending"),
        "PlanDecision": m.PlanDecision(task_id="task", solver_id="generic", budget=budget),
        "AttemptLimits": attempt_limits,
        "RunConfig": m.RunConfig(
            total_budget_units=1000, deadline_monotonic=1000.0, attempt_limits=attempt_limits
        ),
        "SolverContext": p.SolverContext(
            task=task,
            attempt_id="attempt",
            step_index=0,
            budget=budget,
            tools=(spec,),
            llm=LLMDouble(),
        ),
    }


class LLMDouble:
    async def complete(self, request: m.LLMRequest) -> m.LLMReply:
        return m.LLMReply(
            request_id="double",
            text="answer",
            finish="stop",
            usage=m.UsageRecord(
                request_id="double",
                attempt_id="attempt",
                model=request.model,
                status="unknown",
                input_tokens=None,
                output_tokens=None,
            ),
        )


class C01(unittest.TestCase):
    def setUp(self):
        self.dto = examples()

    def invalid(self, name, **changes):
        with self.assertRaises(m.ContractError):
            dataclasses.replace(self.dto[name], **changes)

    def test_all_dtos_have_one_definition_and_frozen_slots_keyword_fields(self):
        classes = {
            name: obj
            for module in (m, p)
            for name, obj in vars(module).items()
            if isinstance(obj, type) and dataclasses.is_dataclass(obj)
        }
        self.assertEqual(set(classes), set(self.dto))
        for name, obj in self.dto.items():
            with self.subTest(dto=name):
                cls = classes[name]
                self.assertIs(type(obj), cls)
                self.assertTrue(cls.__dataclass_params__.frozen)
                self.assertFalse(hasattr(obj, "__dict__"))
                self.assertTrue(all(field.kw_only for field in dataclasses.fields(cls)))
                field = dataclasses.fields(cls)[0].name
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    setattr(obj, field, getattr(obj, field))

    def test_invalid_literals_rejected_on_every_state_field(self):
        for name, obj in self.dto.items():
            for field, hint in typing.get_type_hints(type(obj)).items():
                if typing.get_origin(hint) is typing.Literal:
                    with self.subTest(dto=name, field=field):
                        self.invalid(name, **{field: "unrecognized-state"})

    def test_bool_not_integer_or_finite_number(self):
        for name, obj in self.dto.items():
            for field, hint in typing.get_type_hints(type(obj)).items():
                parts = (
                    typing.get_args(hint) if typing.get_origin(hint) is types.UnionType else (hint,)
                )
                if (int in parts or float in parts) and bool not in parts:
                    for value in (True, False):
                        with self.subTest(dto=name, field=field, value=value):
                            self.invalid(name, **{field: value})

    def test_bool_fields_require_bool(self):
        for name, obj in self.dto.items():
            for field, hint in typing.get_type_hints(type(obj)).items():
                if hint is bool:
                    with self.subTest(dto=name, field=field):
                        self.invalid(name, **{field: 1})

    def test_numbers_cannot_be_negative_strings_or_nonfinite(self):
        for name, obj in self.dto.items():
            for field, hint in typing.get_type_hints(type(obj)).items():
                parts = (
                    typing.get_args(hint) if typing.get_origin(hint) is types.UnionType else (hint,)
                )
                if (int in parts or float in parts) and bool not in parts:
                    values = ("1", float("nan"), float("inf"), float("-inf"))
                    if "deadline" not in field:
                        values += (-1,)
                    for value in values:
                        with self.subTest(dto=name, field=field, value=value):
                            self.invalid(name, **{field: value})

    def test_required_explicit_unknown_and_zero_are_different(self):
        unknown = dataclasses.replace(
            self.dto["UsageRecord"],
            status="unknown",
            input_tokens=0,
            output_tokens=None,
            reasoning_tokens=None,
        )
        self.assertIsNone(unknown.cost_units)
        zero = dataclasses.replace(unknown, status="exact", output_tokens=0)
        self.assertEqual(zero.cost_units, 0)
        self.invalid("UsageRecord", input_tokens=None)
        self.invalid("UsageRecord", output_tokens=None)
        with self.assertRaises(TypeError):
            m.UsageRecord(request_id="r", attempt_id="a", model=MODEL, status="unknown")
        offer = self.dto["SolverOffer"]
        self.assertIsNone(offer.expected_cost_units)
        self.assertIsNone(offer.success_probability)
        self.assertIsNone(offer.expected_duration_s)
        self.assertEqual(dataclasses.replace(offer, expected_cost_units=0).expected_cost_units, 0)
        self.assertEqual(dataclasses.replace(offer, success_probability=0).success_probability, 0)

    def test_usage_cost_reasoning_once_and_overdraw_is_not_hidden(self):
        self.assertEqual(self.dto["UsageRecord"].cost_units, 520)
        self.assertEqual(self.dto["BudgetLease"].reserved_units, 720)
        self.assertEqual(self.dto["BudgetSnapshot"].available_units, 480)
        overdrawn = dataclasses.replace(self.dto["BudgetSnapshot"], spent_units=1500)
        self.assertEqual(overdrawn.spent_units, 1500)
        self.assertEqual(overdrawn.available_units, 0)
        self.invalid("UsageRecord", reasoning_tokens=41)

    def test_lease_states_and_usage_identity(self):
        lease = self.dto["BudgetLease"]
        usage = self.dto["UsageRecord"]
        for state in ("reserved", "dispatched", "released"):
            dataclasses.replace(lease, state=state)
            self.invalid("BudgetLease", state=state, usage=usage)
        for state, record in (
            ("settled", usage),
            (
                "uncertain",
                dataclasses.replace(usage, status="unknown", input_tokens=None, output_tokens=None),
            ),
        ):
            dataclasses.replace(lease, state=state, usage=record)
            self.invalid("BudgetLease", state=state)
            self.invalid(
                "BudgetLease", state=state, usage=dataclasses.replace(record, request_id="x")
            )
        self.invalid("BudgetLease", state="uncertain", usage=usage)

    def test_ids_hashes_and_unknown_error_codes(self):
        for value in ("", "a" * 129, "a\n", "a\x7f", 123):
            with self.subTest(value=value):
                self.invalid("TaskSpec", task_id=value)
        for value in ("", "A" * 64, "g" * 64, "a" * 63, None):
            with self.subTest(hash=value):
                self.invalid("ArtifactRef", sha256=value)
        self.invalid("ErrorInfo", message="a" * 2049)
        error = self.dto["ErrorInfo"]
        failure = m.OperationFailed(error)
        self.assertIs(failure.error, error)
        self.assertEqual(str(failure), "diagnostic")
        self.assertIsInstance(m.ContractError(), ValueError)

    def test_raw_relative_paths_are_checked_before_normalization(self):
        invalid = (
            "",
            ".",
            "..",
            "a/../b",
            "a/./b",
            "./a",
            "a//b",
            "a/",
            "/a",
            "C:/a",
            "C:a",
            "//host/share",
            "a\\b",
            "a\x00b",
        )
        for raw in invalid:
            with self.subTest(path=repr(raw)), self.assertRaises(m.ContractError):
                m.parse_rel_path(raw)
        self.assertEqual(
            m.parse_rel_path("новая папка/file.txt"), PurePosixPath("новая папка/file.txt")
        )
        self.assertEqual(m.parse_rel_path(".", allow_directory=True), PurePosixPath("."))
        self.invalid("ArtifactRef", path=PurePosixPath("."))
        self.invalid("ArtifactRef", path=PurePosixPath("../secret"))
        self.invalid("ArtifactRef", path="a.txt")

    def test_absolute_os_paths_and_duplicate_files(self):
        for name, field in (
            ("TaskSpec", "task_root"),
            ("TaskSpec", "work_root"),
            ("ToolContext", "workspace_root"),
            ("AttemptResult", "workspace_root"),
            ("PublicationPlan", "answer_path"),
            ("LoadIssue", "task_root"),
        ):
            self.invalid(name, **{field: Path("relative")})
        artifact = self.dto["ArtifactRef"]
        self.invalid("TaskSpec", artifacts=(artifact, artifact))
        self.invalid("ToolResult", artifacts=(artifact, artifact))
        self.invalid("ToolContext", protected_paths=(artifact.path, artifact.path))
        self.invalid("AttemptResult", changes=(self.dto["FileChange"],) * 2)

    def test_collections_require_tuples(self):
        for name, obj in self.dto.items():
            for field in dataclasses.fields(obj):
                if isinstance(getattr(obj, field.name), tuple):
                    with self.subTest(dto=name, field=field.name):
                        self.invalid(name, **{field.name: list(getattr(obj, field.name))})

    def test_json_is_copied_and_deeply_immutable_but_serializable(self):
        raw = {"nested": [{"flag": True, "count": 1, "ratio": 0.5, "nil": None}]}
        call = m.ToolCall(tool_name="read", arguments=raw)
        raw["nested"][0]["count"] = 99
        self.assertEqual(call.arguments["nested"][0]["count"], 1)
        for mutation in (
            lambda: call.arguments.update({"x": 1}),
            lambda: call.arguments.__setitem__("x", 1),
            lambda: call.arguments.pop("nested"),
            lambda: call.arguments.clear(),
            lambda: call.arguments.__ior__({"x": 1}),
            lambda: call.arguments.__init__({"x": 1}),
            lambda: call.arguments["nested"].append(1),
            lambda: call.arguments["nested"].__setitem__(0, 1),
            lambda: call.arguments["nested"].__iadd__([1]),
            lambda: call.arguments["nested"].__imul__(2),
            lambda: call.arguments["nested"].__init__([1]),
            lambda: call.arguments["nested"][0].__setitem__("count", 3),
        ):
            with self.subTest(mutation=mutation), self.assertRaises(m.ContractError):
                mutation()
        self.assertEqual(json.loads(json.dumps(call.arguments))["nested"][0]["count"], 1)
        self.assertEqual(dataclasses.asdict(call)["arguments"], call.arguments)
        result = m.ToolResult(call_id="r", tool_name="read", status="ok", data=raw)
        raw["nested"].clear()
        self.assertEqual(len(result.data["nested"]), 1)
        spec = dataclasses.replace(self.dto["ToolSpec"], input_schema={"enum": [1, 2]})
        with self.assertRaises(m.ContractError):
            spec.input_schema["enum"].append(3)

    def test_non_json_cycles_remote_refs_and_callable_schema_rejected(self):
        cycle = []
        cycle.append(cycle)
        for bad in (
            float("nan"),
            float("inf"),
            (1, 2),
            {1: "x"},
            {"x": object()},
            {"x": lambda: 1},
            cycle,
        ):
            with self.subTest(json=type(bad)):
                self.invalid("ToolResult", data=bad)
        self.invalid("ToolCall", arguments=[])
        self.invalid("ToolSpec", input_schema={"properties": {"x": {"$ref": "https://x/schema"}}})
        dataclasses.replace(self.dto["ToolSpec"], input_schema={"$ref": "#/$defs/local"})

    def test_budget_profiles_and_deterministic_attempt(self):
        self.assertEqual(
            self.dto["ProviderProfile"],
            m.ProviderProfile(
                reasoning_mode="unverified",
                input_bound_verified=False,
                total_output_cap_verified=False,
            ),
        )
        for name in ("AttemptBudget", "AttemptLimits"):
            self.invalid(name, allowed_models=())
            self.invalid(name, allowed_models=(MODEL, MODEL))
            self.invalid(name, max_steps=0)
            dataclasses.replace(
                self.dto[name], max_cost_units=0, max_llm_calls=0, allowed_models=()
            )
        for count in (0, 6):
            self.invalid("RunConfig", max_parallel_tasks=count)
        self.assertEqual(self.dto["RunConfig"].mode, "auto")
        self.assertEqual(self.dto["RunConfig"].accounting_mode, "strict")
        self.invalid("AttemptLimits", max_duration_s=0)
        self.invalid("AttemptLimits", max_attempts_per_task=0)
        self.invalid("SolverOffer", success_probability=1.1)
        self.invalid("SolverOffer", applicability="no", reason="")

    def test_tool_reply_and_step_consistency(self):
        error = self.dto["ErrorInfo"]
        self.invalid("ToolSpec", available=False)
        dataclasses.replace(
            self.dto["ToolSpec"], available=False, unavailable_reason="No isolation"
        )
        self.invalid("ToolResult", error=error)
        self.invalid("ToolResult", status="error")
        self.invalid("ToolResult", status="error", error=error, data="unexpected")
        dataclasses.replace(self.dto["ToolResult"], status="error", error=error, data=None)
        self.invalid("LLMReply", request_id="different")
        self.invalid("LLMRequest", messages=())
        self.invalid("LLMRequest", max_output_tokens=0)
        self.invalid(
            "StepRecord", result=dataclasses.replace(self.dto["ToolResult"], tool_name="other")
        )
        self.invalid("SolverContext", step_index=3)
        self.invalid("SolverContext", tools=(self.dto["ToolSpec"],) * 2)
        self.invalid("SolverContext", llm=object())

    def test_candidate_bytes_and_publication_states(self):
        self.assertEqual(self.dto["CandidateAnswer"].text.encode("utf-8"), b"  abc\n")
        self.assertEqual(m.CandidateAnswer(text="").text, "")
        self.invalid("CandidateAnswer", text="x\x00y")
        self.invalid("CandidateAnswer", text="\ud800")
        self.invalid("AttemptResult", candidate=None)
        self.invalid("AttemptResult", snapshot_sha256=None)
        for status in (
            "no_progress",
            "unsupported",
            "blocked_by_evidence",
            "budget_exhausted",
            "deadline",
            "tool_error",
            "provider_error",
            "cancelled",
            "internal_error",
        ):
            with self.subTest(status=status):
                error = self.dto["ErrorInfo"] if status.endswith("error") else None
                failed = dataclasses.replace(
                    self.dto["AttemptResult"],
                    status=status,
                    candidate=None,
                    snapshot_sha256=None,
                    error=error,
                    reason="diagnostic",
                )
                with self.assertRaises(m.ContractError):
                    dataclasses.replace(failed, candidate=self.dto["CandidateAnswer"])
                with self.assertRaises(m.ContractError):
                    dataclasses.replace(failed, changes=(self.dto["FileChange"],))
                with self.assertRaises(m.ContractError):
                    dataclasses.replace(failed, reason="")
                if status.endswith("error"):
                    with self.assertRaises(m.ContractError):
                        dataclasses.replace(failed, error=None)
        self.invalid("FileChange", sha256=None)
        self.invalid("FileChange", operation="delete")
        dataclasses.replace(self.dto["FileChange"], operation="delete", sha256=None)
        self.invalid("SubmissionReceipt", answer_path=None)
        self.invalid("SubmissionReceipt", answer_sha256=None)
        self.invalid("SubmissionReceipt", error=self.dto["ErrorInfo"])
        for status in ("failed", "conflict"):
            self.invalid("SubmissionReceipt", status=status)
            dataclasses.replace(
                self.dto["SubmissionReceipt"],
                status=status,
                answer_path=None,
                answer_sha256=None,
                error=self.dto["ErrorInfo"],
            )

    def test_validation_aggregation_is_failed_then_inconclusive_then_passed(self):
        check = self.dto["CheckResult"]
        self.invalid("ValidationResult", checks=())
        self.invalid("ValidationResult", checks=(check, check))
        for states, expected in (
            (("passed",), "passed"),
            (("inconclusive",), "inconclusive"),
            (("failed",), "failed"),
            (("failed", "inconclusive"), "failed"),
            (("passed", "inconclusive"), "inconclusive"),
        ):
            checks = tuple(
                dataclasses.replace(check, check_id=str(i), status=status)
                for i, status in enumerate(states)
            )
            dataclasses.replace(self.dto["ValidationResult"], checks=checks, status=expected)
            wrong = "failed" if expected == "passed" else "passed"
            self.invalid("ValidationResult", checks=checks, status=wrong)
        self.invalid("AcceptanceDecision", reason="")

    def test_progress_keeps_repeated_attempts_and_rejects_wrong_task(self):
        dataclasses.replace(self.dto["TaskProgress"], attempted_solvers=("generic", "generic"))
        self.invalid("TaskProgress", task_id="other", last_result=self.dto["AttemptResult"])
        self.invalid("LoadReport", tasks=(self.dto["TaskSpec"],) * 2)

    def test_sync_async_port_boundaries(self):
        expected = {
            "TaskLoader": {"load"},
            "TaskAdapter": {"load_task"},
            "SolverRegistry": set(),
            "Solver": {"step"},
            "Planner": set(),
            "BudgetController": set(),
            "LLMGateway": set(),
            "LLMSession": {"complete"},
            "ToolExecutor": {"execute"},
            "AttemptEngine": {"run_attempt"},
            "Validator": {"validate"},
            "AcceptancePolicy": set(),
            "SubmissionLayout": set(),
            "Publisher": {"publish"},
        }
        protocols = {
            name: cls
            for name, cls in vars(p).items()
            if isinstance(cls, type) and getattr(cls, "_is_protocol", False) and name != "Protocol"
        }
        self.assertEqual(set(protocols), set(expected))
        for name, cls in protocols.items():
            actual = {
                method for method, obj in vars(cls).items() if inspect.iscoroutinefunction(obj)
            }
            self.assertEqual(actual, expected[name], name)

    def test_import_is_stdlib_only_without_site_packages_or_network(self):
        script = """
import sys
def audit(event, args):
    if event.startswith("socket."):
        raise AssertionError("network use in contracts")
sys.addaudithook(audit)
from nsu_budget_agent.contracts import models
assert "nsu_budget_agent.contracts.ports" not in sys.modules
from nsu_budget_agent.contracts import ports
assert "openjiuwen" not in sys.modules
assert "pydantic" not in sys.modules
"""
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
        completed = subprocess.run(
            [sys.executable, "-S", "-c", script],
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        for filename in ("models.py", "ports.py"):
            tree = ast.parse(
                (ROOT / "src/nsu_budget_agent/contracts" / filename).read_text("utf-8")
            )
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertIn(alias.name.split(".")[0], sys.stdlib_module_names)
                elif isinstance(node, ast.ImportFrom) and not node.level:
                    self.assertIn(node.module.split(".")[0], sys.stdlib_module_names)

    def test_b_can_use_session_double_without_sdk(self):
        ctx = self.dto["SolverContext"]
        reply = asyncio.run(ctx.llm.complete(self.dto["LLMRequest"]))
        self.assertEqual(reply.text, "answer")
        self.assertIsNone(reply.usage.cost_units)


if __name__ == "__main__":
    unittest.main()
