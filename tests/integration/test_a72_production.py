"""Standard CLI + strict A7.1 wiring; every provider response is offline scripted."""

import contextlib
import io
import json
import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import httpx
from tests.unit.test_a35_provider import ENV, MODELS, mock_compatibility

from nsu_budget_agent.contracts.models import ProviderProfile
from nsu_budget_agent.orchestration.cli import main
from nsu_budget_agent.solvers import composition


class ProductionActivationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
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
        self.environment = {k: v for k, v in os.environ.items() if not k.startswith("LLM_")}
        self.environment.update(BUDGET="100000", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
        self.calls = []
        guard = patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            side_effect=AssertionError("live NeuralDeep forbidden"),
        )
        guard.start()
        self.addCleanup(guard.stop)

    def task(self, instruction, *, csv=False):
        root = self.tasks / "one"
        root.mkdir()
        (root / "task.json").write_text(
            json.dumps(
                {
                    "task_id": "one",
                    "instruction": instruction,
                    "reward": 10,
                    "type": "data_analysis" if csv else "math",
                    "check": "exact_int",
                }
            ),
            encoding="utf-8",
        )
        if csv:
            (root / "sales.csv").write_text(
                "region,status,amount\nwest,paid,20\nwest,cancelled,10\neast,paid,99\n",
                encoding="utf-8",
            )
        return root

    def invoke(self, *, options=()):
        output = io.StringIO()
        with (
            patch.dict(os.environ, self.environment, clear=True),
            contextlib.redirect_stdout(output),
        ):
            code = main(
                ["--tasks-dir", str(self.tasks), "--staging-dir", str(self.staging), *options]
            )
        summaries = [
            json.loads(line)
            for line in output.getvalue().splitlines()
            if line.startswith('{"tasks":')
        ]
        return code, summaries[-1] if summaries else None, output.getvalue()

    def configure(self):
        self.environment["LLM_MODE"] = "strict"
        self.environment.update(ENV)
        self.environment.update(
            LLM_MODEL="qwen3.6-35b-a3b-noreason",
            LLM_COMPATIBILITY_ADAPTER="tests.fixture_only:compatibility",
        )

    def test_default_keyed_cli_competition_needs_no_adapter(self):
        from tests.integration.test_a71_router import direct, review

        root = self.task("Give the result of adding eleven and four.")
        self.environment.update(ENV)
        original = composition.build_competition

        def configured(ledger, config, **kwargs):
            def deliver(request):
                self.assertGreater(ledger.snapshot().held_units, 0)
                wire = json.loads(request.content)
                self.calls.append(wire)
                role = json.loads(wire["messages"][-1]["content"])["role"]
                value = direct("15") if role == "solve" else review()
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"content": json.dumps(value)}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 120,
                            "completion_tokens": 48,
                            "total_tokens": 168,
                            "completion_tokens_details": {"reasoning_tokens": 0},
                        },
                    },
                )

            kwargs["transport"] = httpx.MockTransport(deliver)
            return original(ledger, config, **kwargs)

        with (
            patch.object(composition, "build_competition", side_effect=configured),
            patch(
                "nsu_budget_agent.llm.compatibility.load_compatibility",
                side_effect=AssertionError("competition must not load an adapter"),
            ),
        ):
            code, report, _ = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual((root / "answer.txt").read_text(), "15")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(report["llm_mode"], "competition")
        self.assertEqual(report["accounting_mode"], "best_effort")
        self.assertEqual(report["spent_units"], 1200)
        self.assertEqual(report["limit_units"], 1_000_000)

    def test_explicit_strict_keyed_cli_without_adapter_never_dispatches(self):
        root = self.task("Give the result of adding eleven and four.")
        self.environment.update(ENV)
        code, report, _ = self.invoke(options=("--llm-mode", "strict"))
        self.assertEqual(code, 2)
        self.assertIsNone(report)
        self.assertFalse((root / "answer.txt").exists())

    def test_standard_cli_without_key_is_offline_and_does_not_load_adapter(self):
        root = self.task("Compute 2+3.")
        self.environment.update(LLM_COMPATIBILITY_ADAPTER="do_not_import:this", LLM_API_KEY=" ")
        with patch(
            "nsu_budget_agent.llm.compatibility.load_compatibility",
            side_effect=AssertionError("no adapter without a key"),
        ):
            code, report, _ = self.invoke()
        self.assertEqual(code, 0)
        self.assertEqual((root / "answer.txt").read_text(), "5")
        self.assertEqual(report["spent_units"], 0)
        self.assertEqual(report["held_units"], 0)
        self.assertEqual(report["decision_routes"], {})

    def test_standard_cli_valid_configuration_activates_router_and_accounts_mock_usage(self):
        root = self.task("Total paid west revenue in sales.csv.", csv=True)
        self.configure()
        adapter = replace(mock_compatibility(), input_bound=lambda request, body: 4096)
        original = composition.build_with_router

        def configured(ledger, config, **kwargs):
            def handler(request):
                self.assertGreater(ledger.snapshot().held_units, 0)
                wire = json.loads(request.content)
                self.calls.append(wire)
                self.assertEqual(wire["model"], self.environment["LLM_MODEL"])
                payload = json.loads(wire["messages"][-1]["content"])
                if payload["role"] == "solve":
                    action = {
                        "kind": "tool_plan",
                        "tool_name": "query_csv",
                        "arguments": {
                            "path": "sales.csv",
                            "aggregate": "sum",
                            "columns": ["amount"],
                            "filters": [
                                {"column": "region", "op": "eq", "value": "west"},
                                {"column": "status", "op": "eq", "value": "paid"},
                            ],
                        },
                    }
                else:
                    self.assertEqual(payload["proposal"]["answer"], "20")
                    action = {
                        "verdict": "accept",
                        "reason": "both filters covered",
                        "next_action": "publish",
                        "missing_conditions": [],
                    }
                return httpx.Response(
                    200,
                    json={
                        "choices": [
                            {"message": {"content": json.dumps(action)}, "finish_reason": "stop"}
                        ],
                        "usage": {
                            "prompt_tokens": 120,
                            "completion_tokens": 48,
                            "completion_tokens_details": {"reasoning_tokens": 0},
                        },
                    },
                )

            kwargs["transport"] = httpx.MockTransport(handler)
            return original(ledger, config, **kwargs)

        with (
            patch(
                "nsu_budget_agent.llm.compatibility.load_compatibility", return_value=adapter
            ) as load,
            patch.object(composition, "build_with_router", side_effect=configured) as build_router,
        ):
            code, report, _ = self.invoke()  # No --components and no injected CLI factory.
        self.assertEqual(code, 0)
        load.assert_called_once_with(self.environment["LLM_COMPATIBILITY_ADAPTER"])
        self.assertEqual(build_router.call_count, 1)
        self.assertEqual((root / "answer.txt").read_text(), "20")
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(report["limit_units"], 1_000_000)
        self.assertEqual(report["spent_units"], 1200)
        self.assertEqual(report["held_units"], 0)
        self.assertEqual(report["decision_routes"]["one"]["state"], "PUBLISH")
        self.assertFalse(report["decision_routes"]["one"]["independent_semantics"])
        self.assertEqual(report["workflow_invocations"], 2)

    def test_key_without_full_configuration_refuses_before_transport(self):
        self.task("Compute 2+3.")
        for missing in ("LLM_BASE_URL", "LLM_COMPATIBILITY_ADAPTER"):
            self.configure()
            self.environment.pop(missing)
            with (
                self.subTest(missing=missing),
                patch(
                    "httpx.AsyncHTTPTransport",
                    side_effect=AssertionError("must fail before transport"),
                ),
            ):
                code, report, output = self.invoke()
            self.assertEqual(code, 2)
            self.assertIsNone(report)
            self.assertNotIn(ENV["LLM_API_KEY"], output)

    def test_unverified_profile_never_activates_and_is_not_promoted(self):
        root = self.task("Compute 2+3.")
        self.configure()
        for field in ("input", "output", "usage"):
            profile = ProviderProfile(
                reasoning_mode="unverified" if field == "usage" else "included",
                input_bound_verified=field != "input",
                total_output_cap_verified=field != "output",
            )
            adapter = replace(mock_compatibility(), profiles={model: profile for model in MODELS})
            with (
                self.subTest(field=field),
                patch(
                    "nsu_budget_agent.llm.compatibility.load_compatibility", return_value=adapter
                ),
                patch(
                    "httpx.AsyncHTTPTransport", side_effect=AssertionError("unverified transport")
                ),
            ):
                code, report, _ = self.invoke()
            self.assertEqual(code, 2)
            self.assertIsNone(report)
            self.assertEqual(adapter.profiles[self.environment["LLM_MODEL"]], profile)
        self.assertFalse((root / "answer.txt").exists())

    def test_explicit_offline_factory_works_with_untrusted_credentials_present(self):
        root = self.task("Compute 2+3.")
        self.configure()
        code, report, _ = self.invoke(
            options=("--components", "nsu_budget_agent.solvers.composition:build")
        )
        self.assertEqual(code, 0)
        self.assertEqual((root / "answer.txt").read_text(), "5")
        self.assertEqual(report["spent_units"], 0)
