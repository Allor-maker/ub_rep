"""Only MockTransport profiles are verified here; real provider stays unconfirmed."""

import contextlib
import io
import json
import os
import unittest
from dataclasses import replace
from typing import get_args
from unittest.mock import patch

import httpx

from nsu_budget_agent.contracts.models import ContractError, ModelId, ProviderProfile
from nsu_budget_agent.llm.compatibility import ProviderCompatibility, load_compatibility
from nsu_budget_agent.llm.provider_smoke import main, run_smoke

MODELS = get_args(ModelId)
ENV = {"LLM_BASE_URL": "https://proxy.example.invalid/v1", "LLM_API_KEY": "private-test-key"}


def mock_compatibility(reasoning="included", parameter="max_tokens"):
    return ProviderCompatibility(
        input_bound=lambda request, body: 128,
        profiles={
            model: ProviderProfile(
                reasoning_mode=reasoning, input_bound_verified=True, total_output_cap_verified=True
            )
            for model in MODELS
        },
        evidence="MockTransport fixture only; not valid for NeuralDeep",
        output_token_parameter=parameter,
    )


def payload(completion=5, reasoning=2):
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": "PRIVATE CONTENT", "reasoning_content": "PRIVATE REASONING"},
            }
        ],
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": completion,
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }


class ProviderSmokeTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        guard = patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            side_effect=AssertionError("live API forbidden"),
        )
        guard.start()
        self.addCleanup(guard.stop)

    async def smoke(self, handler, **options):
        config = dict(
            live=True,
            budget_units=10_000,
            compatibility=mock_compatibility(),
            environ=ENV,
            transport=httpx.MockTransport(handler),
        )
        config.update(options)
        return await run_smoke(**config)

    async def test_four_models_two_caps_usage_reasoning_no_profile_promotion(self):
        calls = []

        async def handler(request):
            body = json.loads(request.content)
            calls.append(body)
            self.assertEqual(request.headers["authorization"], "Bearer " + ENV["LLM_API_KEY"])
            self.assertFalse(body["stream"])
            self.assertEqual(body["n"], 1)
            return httpx.Response(200, json=payload())

        report = await self.smoke(handler)
        self.assertEqual(
            [(c["model"], c["max_tokens"]) for c in calls],
            [(m, cap) for m in MODELS for cap in (8, 64)],
        )
        self.assertEqual(report["spent_units"], 8 * (20 + 10 * 5))
        self.assertEqual(report["held_units"], 0)
        self.assertTrue(report["strict"])
        self.assertFalse(report["profile_promoted"])
        self.assertTrue(all(row["output_cap_observed_within"] for row in report["observations"]))
        self.assertTrue(report["observations"][0]["observed"]["reasoning_text_present"])
        for secret in (
            ENV["LLM_API_KEY"],
            ENV["LLM_BASE_URL"],
            "PRIVATE CONTENT",
            "PRIVATE REASONING",
        ):
            self.assertNotIn(secret, json.dumps(report))

    async def test_separate_reasoning_and_explicit_completion_parameter(self):
        def handler(request):
            self.assertIn("max_completion_tokens", json.loads(request.content))
            self.assertNotIn("max_tokens", json.loads(request.content))
            return httpx.Response(200, json=payload())

        report = await self.smoke(
            handler,
            models=(MODELS[0],),
            compatibility=mock_compatibility("separate", "max_completion_tokens"),
        )
        self.assertEqual(report["spent_units"], 2 * (20 + 10 * (5 + 2)))
        self.assertEqual(report["observations"][0]["output_tokens"], 7)

    async def test_explicit_best_effort_keeps_all_reservations_and_never_promotes(self):
        report = await self.smoke(
            lambda request: httpx.Response(200, json=payload()),
            compatibility=None,
            best_effort_diagnostic=True,
            input_bound_tokens=128,
        )
        self.assertEqual(len(report["observations"]), 8)
        self.assertFalse(report["strict"])
        self.assertTrue(report["completed_samples"])
        self.assertFalse(report["profile_promoted"])
        self.assertEqual(report["spent_units"], 0)
        self.assertEqual(report["held_units"], 4 * (128 + 80 + 128 + 640))
        self.assertTrue(all(row["usage_status"] == "unknown" for row in report["observations"]))
        self.assertTrue(
            all(row["output_cap_observed_within"] is None for row in report["observations"])
        )

    async def test_diagnostic_requires_explicit_estimate_and_stops_on_missing_usage(self):
        for changes in ({"input_bound_tokens": None}, {"input_bound_tokens": 0}):
            with self.assertRaises(ContractError):
                await self.smoke(
                    lambda request: self.fail("must not send"),
                    compatibility=None,
                    best_effort_diagnostic=True,
                    **changes,
                )
        data = payload()
        del data["usage"]
        report = await self.smoke(
            lambda request: httpx.Response(200, json=data),
            compatibility=None,
            best_effort_diagnostic=True,
            input_bound_tokens=128,
        )
        self.assertEqual(len(report["observations"]), 1)
        self.assertFalse(report["completed_samples"])
        self.assertEqual(report["held_units"], 208)

    async def test_diagnostic_records_ambiguous_reasoning_without_inferring_mode(self):
        report = await self.smoke(
            lambda request: httpx.Response(200, json=payload(completion=8, reasoning=6)),
            models=(MODELS[0],),
            compatibility=None,
            best_effort_diagnostic=True,
            input_bound_tokens=128,
        )
        first = report["observations"][0]
        self.assertTrue(first["cap_consistent_if_included"])
        self.assertFalse(first["cap_consistent_if_separate"])
        self.assertIsNone(first["output_cap_observed_within"])
        self.assertFalse(report["completed_samples"])
        self.assertEqual(len(report["observations"]), 1)
        self.assertEqual(report["spent_units"], 0)

    async def test_all_model_series_stops_on_possible_separate_reasoning_over_cap(self):
        calls = []

        def handler(request):
            cap = json.loads(request.content)["max_tokens"]
            calls.append(cap)
            return httpx.Response(200, json=payload(completion=cap, reasoning=cap))

        report = await self.smoke(
            handler,
            compatibility=None,
            best_effort_diagnostic=True,
            input_bound_tokens=20,
            budget_units=3040,
        )
        self.assertEqual(calls, [8])  # The other seven planned samples are not sent.
        self.assertEqual(report["held_units"], 100)
        self.assertEqual(report["spent_units"], 0)
        self.assertFalse(report["completed_samples"])
        self.assertFalse(report["profile_promoted"])
        self.assertFalse(report["observations"][0]["cap_consistent_if_separate"])

    async def test_streaming_observation_preserves_body_and_redacts_text(self):
        class Stream(httpx.AsyncByteStream):
            async def __aiter__(self):
                raw = json.dumps(payload()).encode()
                yield raw[:30]
                yield raw[30:]

        report = await self.smoke(
            lambda request: httpx.Response(200, stream=Stream()), models=(MODELS[0],)
        )
        self.assertEqual(report["observations"][0]["observed"]["nested_reasoning_tokens"], 2)
        self.assertEqual(report["held_units"], 0)

    async def test_missing_live_key_or_verified_profile_never_sends(self):
        for changes in (
            {"live": False},
            {"environ": {**ENV, "LLM_API_KEY": ""}},
            {"budget_units": 0},
            {
                "compatibility": replace(
                    mock_compatibility(), profiles={m: ProviderProfile() for m in MODELS}
                )
            },
        ):
            with self.subTest(changes=tuple(changes)):
                with self.assertRaises(ContractError):
                    await self.smoke(lambda request: self.fail("must not dispatch"), **changes)

    async def test_cap_violation_halts_without_retry(self):
        calls = []

        def handler(request):
            calls.append(request)
            return httpx.Response(200, json=payload(completion=9))

        report = await self.smoke(handler)
        self.assertEqual(len(calls), 1)
        self.assertEqual(report["observations"][0]["error_code"], "accounting_violation")
        self.assertTrue(report["blocked"])
        self.assertEqual(report["spent_units"], 110)

    async def test_unknown_usage_retains_reserve_and_stops(self):
        data = payload()
        del data["usage"]
        report = await self.smoke(lambda request: httpx.Response(200, json=data))
        self.assertEqual(len(report["observations"]), 1)
        self.assertEqual(report["held_units"], 128 + 10 * 8)
        self.assertEqual(report["spent_units"], 0)

    async def test_timeout_and_http_refusal_are_redacted_no_retry(self):
        def timeout(request):
            raise httpx.ReadTimeout("PRIVATE " + ENV["LLM_API_KEY"])

        for handler in (
            timeout,
            lambda request: httpx.Response(
                429, json={"error": {"message": ENV["LLM_API_KEY"]}}, headers={"Retry-After": "1"}
            ),
        ):
            report = await self.smoke(handler)
            self.assertEqual(len(report["observations"]), 1)
            self.assertEqual(report["held_units"], 208)
            self.assertNotIn(ENV["LLM_API_KEY"], json.dumps(report))


class CompatibilityConfigTests(unittest.TestCase):
    def test_cli_diagnostic_is_explicit_offline_and_never_enables_strict(self):
        output = io.StringIO()
        transport = httpx.MockTransport(lambda request: httpx.Response(200, json=payload()))
        with (
            patch.dict(os.environ, ENV, clear=True),
            contextlib.redirect_stdout(output),
            patch.object(httpx, "AsyncHTTPTransport", return_value=transport),
        ):
            result = main(
                [
                    "--live",
                    "--best-effort-diagnostic",
                    "--input-bound-tokens",
                    "128",
                    "--budget-units",
                    "2000",
                    "--model",
                    MODELS[0],
                ]
            )
        self.assertEqual(result, 0)
        report = json.loads(output.getvalue())
        self.assertFalse(report["strict"])
        self.assertFalse(report["profile_promoted"])
        self.assertTrue(report["completed_samples"])

    def test_cli_requires_flag_key_budget_adapter_without_transport(self):
        for args, env in (
            ([], ENV),
            (["--live", "--budget-units", "1000"], {}),
            (["--live", "--budget-units", "1000"], ENV),
        ):
            output = io.StringIO()
            with (
                patch.dict(os.environ, env, clear=True),
                contextlib.redirect_stdout(output),
                patch.object(
                    httpx, "AsyncHTTPTransport", side_effect=AssertionError("no transport")
                ),
            ):
                self.assertEqual(main(args), 2)
            self.assertNotIn(ENV["LLM_API_KEY"], output.getvalue())

    def test_profiles_are_copied_readonly_and_never_inferred(self):
        source = dict(mock_compatibility().profiles)
        adapter = replace(mock_compatibility(), profiles=source)
        source.clear()
        adapter.require_verified(MODELS)
        with self.assertRaises(TypeError):
            adapter.profiles[MODELS[0]] = ProviderProfile()
        with self.assertRaises(ContractError):
            replace(adapter, evidence="")
        with self.assertRaises(ContractError):
            load_compatibility("task/adapter.py")
