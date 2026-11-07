"""HTTPX MockTransport only: no DNS, sockets, API keys or paid calls."""

import asyncio
import dataclasses
import json
import os
import unittest
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from unittest.mock import AsyncMock, patch

import httpx

from nsu_budget_agent.budget import BudgetLedger
from nsu_budget_agent.contracts.models import (
    AttemptBudget,
    ChatMessage,
    ContractError,
    LLMRequest,
    OperationFailed,
    ProviderProfile,
    ToolLimits,
)
from nsu_budget_agent.llm import NeuralDeepGateway
from nsu_budget_agent.llm.gateway import _retry_after
from nsu_budget_agent.llm.usage import normalize_usage

MODELS = ("qwen3.8-27b", "qwen3.8-27b-noreason", "qwen3.6-35b-a3b", "qwen3.6-35b-a3b-noreason")
ENV = {"LLM_BASE_URL": "https://proxy.example.invalid/v1", "LLM_API_KEY": "test-key-never-live"}
VERIFIED = ProviderProfile(
    reasoning_mode="included", input_bound_verified=True, total_output_cap_verified=True
)


def payload(**changes):
    return {
        "choices": [
            {
                "message": {"content": "  answer\n", "reasoning_content": "PRIVATE REASONING"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 120,
            "completion_tokens": 40,
            "completion_tokens_details": {"reasoning_tokens": 20},
            "prompt_tokens_details": {"cached_tokens": 100},
            "total_tokens": 9999,
        },
        **changes,
    }


class GatewayTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.loop = asyncio.get_running_loop()
        self.deadline = self.loop.time() + 60
        self.ledger = BudgetLedger(limit_units=20000, deadline_monotonic=self.deadline)
        self.budget = AttemptBudget(
            max_cost_units=20000,
            max_llm_calls=8,
            max_steps=8,
            deadline_monotonic=self.deadline,
            allowed_models=MODELS,
            tool_limits=ToolLimits(
                timeout_s=1,
                max_read_bytes=100,
                max_output_bytes=100,
                max_write_bytes=100,
                max_process_memory_bytes=100,
                max_process_count=1,
            ),
        )
        self.events = []
        self.calls = []
        guard = patch.object(
            httpx.AsyncHTTPTransport,
            "handle_async_request",
            side_effect=AssertionError("real network is forbidden in tests"),
        )
        guard.start()
        self.addCleanup(guard.stop)

    def gateway(self, handler=None, *, bypass_rate=True, **options):
        async def handle(request):
            self.calls.append(request)
            # The accounting commit is visible before the HTTP transport is entered.
            self.assertGreater(self.ledger.snapshot().held_units, 0)
            self.assertEqual(self.events[-1].phase, "dispatched")
            if handler is None:
                return httpx.Response(200, json=payload())
            result = handler(request)
            return await result if asyncio.iscoroutine(result) else result

        config = {
            "budget": self.ledger,
            "input_bound": lambda request, body: 120,
            "profiles": {model: VERIFIED for model in MODELS},
            "environ": ENV,
            "event_sink": self.events.append,
            "transport": httpx.MockTransport(handle),
            **options,
        }
        gateway = NeuralDeepGateway(**config)
        self.addAsyncCleanup(gateway.aclose)
        if bypass_rate:
            limiter = patch.object(gateway, "_wait_rate_slot", new=AsyncMock())
            limiter.start()
            self.addCleanup(limiter.stop)
        return gateway

    def request(self, *, model=MODELS[0], max_output_tokens=60):
        return LLMRequest(
            model=model,
            messages=(ChatMessage(role="user", content="PRIVATE PROMPT"),),
            max_output_tokens=max_output_tokens,
        )

    async def fails(self, code, awaitable):
        with self.assertRaises(OperationFailed) as error:
            await awaitable
        self.assertEqual(error.exception.error.code, code)
        return error.exception.error

    async def test_all_models_wire_body_accounting_and_safe_events(self):
        observed = []

        def bound(request, body):
            observed.append(body)
            self.assertEqual(json.loads(body)["model"], request.model)
            return 120

        gateway = self.gateway(input_bound=bound)
        ids = []
        for index, model in enumerate(MODELS):
            reply = await gateway.bind("a", index, self.budget).complete(self.request(model=model))
            ids.append(reply.request_id)
            self.assertRegex(reply.request_id, r"^[0-9a-f]{32}$")
            self.assertEqual(reply.usage.model, model)
            self.assertEqual(reply.text, "  answer\n")
            self.assertEqual(reply.usage.cost_units, 520)  # No cache discount / extra reasoning.
            wire = self.calls[-1]
            self.assertEqual(wire.method, "POST")
            self.assertEqual(str(wire.url), ENV["LLM_BASE_URL"] + "/chat/completions")
            self.assertEqual(wire.headers["authorization"], "Bearer " + ENV["LLM_API_KEY"])
            self.assertEqual(wire.content, observed[-1])
            self.assertEqual(json.loads(wire.content)["max_tokens"], 60)
            self.assertFalse(json.loads(wire.content)["stream"])
            self.assertEqual(json.loads(wire.content)["n"], 1)
        self.assertEqual(len(set(ids)), 4)
        self.assertEqual(self.ledger.snapshot().spent_units, 4 * 520)
        self.assertEqual(self.ledger.snapshot().held_units, 0)
        safe = repr(self.events)
        for secret in (*ENV.values(), "PRIVATE PROMPT", "PRIVATE REASONING", "answer"):
            self.assertNotIn(secret, safe)

    async def test_usage_modes_and_malformed_counters(self):
        for mode, data, exact, output in (
            ("included", {"prompt_tokens": 120, "completion_tokens": 40}, True, 40),
            (
                "separate",
                {
                    "prompt_tokens": 120,
                    "completion_tokens": 40,
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
                True,
                60,
            ),
            ("separate", {"prompt_tokens": 120, "completion_tokens": 40}, False, 40),
            ("unverified", {"prompt_tokens": 120, "completion_tokens": 40}, False, 40),
            ("included", {"prompt_tokens": True, "completion_tokens": 40}, False, 40),
            ("included", {"prompt_tokens": 120, "completion_tokens": -1}, False, None),
            (
                "included",
                {
                    "prompt_tokens": 120,
                    "completion_tokens": 40,
                    "reasoning_tokens": 5,
                    "completion_tokens_details": {"reasoning_tokens": 6},
                },
                False,
                40,
            ),
            (
                "included",
                {
                    "prompt_tokens": 120,
                    "completion_tokens": 40,
                    "completion_tokens_details": {"reasoning_tokens": 50},
                },
                False,
                40,
            ),
        ):
            with self.subTest(mode=mode, data=data):
                record = normalize_usage(
                    {"usage": data},
                    request_id="r",
                    attempt_id="a",
                    model=MODELS[0],
                    profile=ProviderProfile(reasoning_mode=mode),
                    success=True,
                )
                self.assertEqual(record.status, "exact" if exact else "unknown")
                self.assertEqual(record.output_tokens, output)
                self.assertEqual(record.cost_units, 120 + 10 * output if exact else None)

    async def test_separate_reasoning_in_gateway_is_charged_once(self):
        gateway = self.gateway(
            profiles={MODELS[0]: dataclasses.replace(VERIFIED, reasoning_mode="separate")}
        )
        reply = await gateway.bind("a", 0, self.budget).complete(self.request())
        self.assertEqual(reply.usage.output_tokens, 60)
        self.assertEqual(self.ledger.snapshot().spent_units, 720)

    async def test_strict_requires_all_verified_profile_fields(self):
        for profile in (
            ProviderProfile(),
            dataclasses.replace(VERIFIED, input_bound_verified=False),
            dataclasses.replace(VERIFIED, total_output_cap_verified=False),
            dataclasses.replace(VERIFIED, reasoning_mode="unverified"),
        ):
            gateway = self.gateway(profiles={MODELS[0]: profile})
            await self.fails(
                "unsupported", gateway.bind("a", 0, self.budget).complete(self.request())
            )
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(self.ledger.snapshot().calls_used, 0)

    async def test_explicit_best_effort_and_missing_usage_preserve_full_hold(self):
        for body in (payload(), payload(usage=None)):
            gateway = self.gateway(
                lambda request: httpx.Response(200, json=body),
                profiles={},
                accounting_mode="best_effort",
            )
            reply = await gateway.bind("a", len(self.calls), self.budget).complete(self.request())
            self.assertEqual(reply.usage.status, "unknown")
            self.assertIsNone(reply.usage.cost_units)
        self.assertEqual(self.ledger.snapshot().held_units, 1440)
        self.assertEqual(self.ledger.snapshot().spent_units, 0)

    async def test_verified_profile_without_usage_still_retains_reservation(self):
        gateway = self.gateway(lambda request: httpx.Response(200, json=payload(usage=None)))
        reply = await gateway.bind("a", 0, self.budget).complete(self.request())
        self.assertEqual(reply.text, "  answer\n")
        self.assertEqual(reply.usage.status, "unknown")
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertEqual(self.ledger.snapshot().spent_units, 0)

    async def test_one_complete_even_after_error_and_across_rebinding(self):
        gateway = self.gateway(lambda request: httpx.Response(500, json={}))
        first = gateway.bind("a", 0, self.budget)
        self.assertIs(gateway.bind("a", 0, self.budget), first)
        await self.fails("provider_error", first.complete(self.request()))
        with self.assertRaises(ContractError):
            await gateway.bind("a", 0, self.budget).complete(self.request())
        with self.assertRaises(ContractError):
            gateway.bind("a", 0, dataclasses.replace(self.budget, max_llm_calls=2))
        for step in (-1, True, 8):
            with self.assertRaises(ContractError):
                gateway.bind("a", step, self.budget)
        self.assertEqual(len(self.calls), 1)

    async def test_concurrent_calls_on_same_session_send_once(self):
        gateway = self.gateway()
        session = gateway.bind("a", 0, self.budget)
        results = await asyncio.gather(
            session.complete(self.request()),
            session.complete(self.request()),
            return_exceptions=True,
        )
        self.assertEqual(sum(isinstance(item, ContractError) for item in results), 1)
        self.assertEqual(len(self.calls), 1)

    async def test_budget_exhaustion_and_disallowed_model_never_reach_http(self):
        limited = dataclasses.replace(self.budget, max_cost_units=719)
        gateway = self.gateway()
        await self.fails(
            "budget_exhausted", gateway.bind("small", 0, limited).complete(self.request())
        )
        singleton = dataclasses.replace(self.budget, allowed_models=(MODELS[1],))
        with self.assertRaises(ContractError):
            await gateway.bind("other", 0, singleton).complete(self.request())
        self.assertEqual(len(self.calls), 0)
        self.assertEqual(self.ledger.snapshot().held_units, 0)

    async def test_malformed_envelopes_still_account_valid_usage(self):
        for body in (
            payload(choices=[]),
            payload(choices=[{"message": {"content": None}}]),
            payload(choices=[{"message": {"content": "x", "tool_calls": [{}]}}]),
        ):
            gateway = self.gateway(lambda request: httpx.Response(200, json=body))
            await self.fails(
                "invalid_output",
                gateway.bind("a", len(self.calls), self.budget).complete(self.request()),
            )
        self.assertEqual(self.ledger.snapshot().spent_units, 3 * 520)
        self.assertEqual(self.ledger.snapshot().held_units, 0)

    async def test_invalid_json_and_oversized_responses_keep_unknown_hold(self):
        for content, options in (
            (b"not JSON", {}),
            (b'{"usage":{},"usage":{}}', {}),
            (b'{"usage":NaN}', {}),
            (b" " * 200, {"max_response_bytes": 100}),
        ):
            gateway = self.gateway(lambda request: httpx.Response(200, content=content), **options)
            await self.fails(
                "invalid_output",
                gateway.bind("a", len(self.calls), self.budget).complete(self.request()),
            )
        self.assertEqual(self.ledger.snapshot().held_units, 4 * 720)

    async def test_error_statuses_are_sanitized_not_retried_and_not_free(self):
        for status, code, retryable in (
            (400, "provider_error", False),
            (500, "provider_error", True),
            (503, "provider_error", True),
            (429, "rate_limited", True),
            (302, "provider_error", False),
        ):
            body = {
                "error": {"message": ENV["LLM_API_KEY"], "code": "other"},
                "usage": {"prompt_tokens": 0, "completion_tokens": 0},
            }
            gateway = self.gateway(
                lambda request: httpx.Response(
                    status, json=body, headers={"location": "https://never-follow.invalid"}
                )
            )
            error = await self.fails(
                code, gateway.bind("a", len(self.calls), self.budget).complete(self.request())
            )
            self.assertEqual(error.retryable, retryable)
            self.assertNotIn(ENV["LLM_API_KEY"], repr(error))
        self.assertEqual(len(self.calls), 5)
        self.assertEqual(self.ledger.snapshot().held_units, 5 * 720)

    async def test_final_quota_halts_already_queued_and_future_requests(self):
        gateway = self.gateway(
            lambda request: httpx.Response(429, json={"error": {"code": "insufficient_quota"}})
        )
        await self.fails(
            "quota_exhausted", gateway.bind("a", 0, self.budget).complete(self.request())
        )
        await self.fails(
            "quota_exhausted", gateway.bind("a", 1, self.budget).complete(self.request())
        )
        self.assertEqual(len(self.calls), 1)
        self.assertTrue(self.ledger.snapshot().blocked)
        self.assertEqual(self.ledger.snapshot().held_units, 720)

    async def test_auth_failure_halts_without_leaking_provider_message(self):
        gateway = self.gateway(
            lambda request: httpx.Response(401, json={"error": {"message": ENV["LLM_BASE_URL"]}})
        )
        error = await self.fails(
            "denied", gateway.bind("a", 0, self.budget).complete(self.request())
        )
        self.assertNotIn(ENV["LLM_BASE_URL"], repr(error))
        self.assertTrue(self.ledger.snapshot().blocked)

    async def test_quota_failure_prevents_dispatch_of_an_existing_reservation(self):
        started, finish = asyncio.Event(), asyncio.Event()

        async def handler(request):
            started.set()
            await finish.wait()
            return httpx.Response(429, json={"error": {"code": "quota_exhausted"}})

        gateway = self.gateway(handler, max_concurrent_requests=1)
        first = asyncio.create_task(gateway.bind("a", 0, self.budget).complete(self.request()))
        await started.wait()
        second = asyncio.create_task(gateway.bind("b", 0, self.budget).complete(self.request()))
        await asyncio.sleep(0)
        self.assertEqual(self.ledger.snapshot().held_units, 1440)
        finish.set()
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(isinstance(result, OperationFailed) for result in results))
        self.assertEqual([result.error.code for result in results], ["quota_exhausted"] * 2)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertEqual(self.ledger.snapshot().calls_used, 1)

    async def test_length_and_unknown_finish_are_preserved_not_executed(self):
        for index, finish in enumerate(("length", "unexpected")):
            body = payload(
                choices=[{"message": {"content": '{"partial":'}, "finish_reason": finish}]
            )
            gateway = self.gateway(lambda request: httpx.Response(200, json=body))
            reply = await gateway.bind("a", index, self.budget).complete(self.request())
            self.assertEqual(reply.finish, "length" if index == 0 else "other")
            self.assertEqual(reply.text, '{"partial":')

    async def test_http_timeout_after_dispatch_is_uncertain(self):
        def handler(request):
            raise httpx.ConnectTimeout(ENV["LLM_API_KEY"])

        gateway = self.gateway(handler)
        error = await self.fails(
            "timeout", gateway.bind("a", 0, self.budget).complete(self.request())
        )
        self.assertNotIn(ENV["LLM_API_KEY"], repr(error))
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertEqual(self.ledger.snapshot().calls_used, 1)

    async def test_async_deadline_after_dispatch_closes_transport_response_work(self):
        cleaned = asyncio.Event()

        async def handler(request):
            try:
                await asyncio.Event().wait()
            finally:
                cleaned.set()

        gateway = self.gateway(handler, timeout_s=0.01)
        await self.fails("timeout", gateway.bind("a", 0, self.budget).complete(self.request()))
        self.assertTrue(cleaned.is_set())
        self.assertEqual(self.ledger.snapshot().held_units, 720)

    async def test_cancellation_while_reading_response_closes_stream_and_keeps_hold(self):
        started, closed = asyncio.Event(), asyncio.Event()

        class WaitingStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                started.set()
                await asyncio.Event().wait()
                yield b"unreachable"

            async def aclose(self):
                closed.set()

        gateway = self.gateway(lambda request: httpx.Response(200, stream=WaitingStream()))
        task = asyncio.create_task(gateway.bind("a", 0, self.budget).complete(self.request()))
        await started.wait()
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        self.assertTrue(closed.is_set())
        self.assertEqual(self.ledger.snapshot().held_units, 720)

    async def test_transport_errors_are_sanitized_and_not_retried(self):
        def handler(request):
            raise httpx.ReadError(ENV["LLM_API_KEY"] + ENV["LLM_BASE_URL"])

        gateway = self.gateway(handler)
        error = await self.fails(
            "provider_error", gateway.bind("a", 0, self.budget).complete(self.request())
        )
        self.assertTrue(error.retryable)
        self.assertNotIn(ENV["LLM_API_KEY"], repr(error))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.ledger.snapshot().held_units, 720)

    async def test_timeout_in_rate_queue_releases_unsent_reservation(self):
        gateway = self.gateway(bypass_rate=False, timeout_s=0.01)
        gateway._next_send_at = self.loop.time() + 60
        await self.fails("timeout", gateway.bind("a", 0, self.budget).complete(self.request()))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger.snapshot().held_units, 0)
        self.assertEqual(self.ledger.snapshot().calls_used, 0)

    async def test_invalid_url_is_unsent_and_releases_reservation(self):
        gateway = self.gateway(environ={"LLM_BASE_URL": "https://\ud800", "LLM_API_KEY": "key"})
        error = await self.fails(
            "invalid_input", gateway.bind("a", 0, self.budget).complete(self.request())
        )
        self.assertNotIn("key", repr(error))
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger.snapshot().held_units, 0)
        self.assertEqual(self.ledger.snapshot().calls_used, 0)
        self.assertNotIn("dispatched", [event.phase for event in self.events])

    async def test_concurrency_limit_cancellation_before_and_after_dispatch(self):
        started = asyncio.Event()

        async def handler(request):
            started.set()
            await asyncio.Event().wait()

        gateway = self.gateway(handler, max_concurrent_requests=1)
        first = asyncio.create_task(gateway.bind("a", 0, self.budget).complete(self.request()))
        await started.wait()
        second = asyncio.create_task(gateway.bind("b", 0, self.budget).complete(self.request()))
        await asyncio.sleep(0)
        self.assertEqual(self.ledger.snapshot().held_units, 1440)
        self.assertEqual(len(self.calls), 1)
        second.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await second
        self.assertIsNone(self.events[-1].usage_status)
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        first.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first
        self.assertEqual(self.events[-1].usage_status, "unknown")
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertEqual(self.ledger.snapshot().calls_used, 1)

    async def test_two_inflight_and_global_budget_competition(self):
        self.ledger = BudgetLedger(limit_units=1440, deadline_monotonic=self.deadline)
        two_started = asyncio.Event()
        finish = asyncio.Event()
        active = peak = 0

        async def handler(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            if active == 2:
                two_started.set()
            try:
                await finish.wait()
                return httpx.Response(200, json=payload())
            finally:
                active -= 1

        gateway = self.gateway(handler, max_concurrent_requests=2)
        tasks = [
            asyncio.create_task(gateway.bind(key, 0, self.budget).complete(self.request()))
            for key in ("a", "b")
        ]
        await two_started.wait()
        await self.fails(
            "budget_exhausted", gateway.bind("c", 0, self.budget).complete(self.request())
        )
        finish.set()
        await asyncio.gather(*tasks)
        self.assertEqual(peak, 2)
        self.assertEqual(self.ledger.snapshot().spent_units, 1040)

    async def test_rate_spacing_and_retry_after_cooldown(self):
        gateway = self.gateway(bypass_rate=False)
        now = [self.loop.time()]
        waits = []

        async def sleep(delay):
            waits.append(delay)
            now[0] += delay

        with (
            patch.object(self.loop, "time", side_effect=lambda: now[0]),
            patch("nsu_budget_agent.llm.gateway.asyncio.sleep", new=sleep),
        ):
            await gateway._wait_rate_slot()
            await gateway._wait_rate_slot()
            self.assertEqual(waits, [1.0])
            gateway._cooldown_until = now[0] + 7
            await gateway._wait_rate_slot()
            self.assertAlmostEqual(waits[-1], 7)
        throttled = self.gateway(
            lambda request: httpx.Response(429, json={}, headers={"Retry-After": "120"}),
            bypass_rate=False,
            timeout_s=0.01,
        )
        error = await self.fails(
            "rate_limited", throttled.bind("a", 0, self.budget).complete(self.request())
        )
        self.assertEqual(error.retry_after_s, 120)
        self.assertGreater(throttled._cooldown_until, self.loop.time() + 119)
        await self.fails("timeout", throttled.bind("a", 1, self.budget).complete(self.request()))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.ledger.snapshot().held_units, 720)

    async def test_retry_after_dates_invalid_headers_and_oversized_429(self):
        self.assertIsNone(_retry_after("NaN"))
        self.assertIsNone(_retry_after("Infinity"))
        self.assertIsNone(_retry_after("invalid"))
        self.assertEqual(_retry_after("-1"), 0)
        date = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=60), usegmt=True)
        self.assertGreater(_retry_after(date), 58)
        self.assertLessEqual(_retry_after(date), 60)
        gateway = self.gateway(
            lambda request: httpx.Response(429, content=b"x" * 200, headers={"Retry-After": "2"}),
            max_response_bytes=100,
        )
        error = await self.fails(
            "rate_limited", gateway.bind("a", 0, self.budget).complete(self.request())
        )
        self.assertEqual(error.retry_after_s, 2)
        self.assertEqual(self.ledger.snapshot().held_units, 720)

    async def test_429_slow_body_blocks_waiting_rate_slot(self):
        allow_headers, reading_body = asyncio.Event(), asyncio.Event()

        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                reading_body.set()
                await asyncio.Event().wait()
                yield b"{}"

        async def handler(request):
            await allow_headers.wait()
            return httpx.Response(429, headers={"Retry-After": "60"}, stream=SlowStream())

        gateway = self.gateway(handler, bypass_rate=False, timeout_s=5)
        first = asyncio.create_task(gateway.bind("a", 0, self.budget).complete(self.request()))
        while not self.calls:
            await asyncio.sleep(0)
        second = asyncio.create_task(gateway.bind("b", 0, self.budget).complete(self.request()))
        await asyncio.sleep(0)
        self.assertTrue(gateway._send_lock.locked())
        allow_headers.set()
        await reading_body.wait()
        self.assertGreater(gateway._cooldown_until, self.loop.time() + 59)
        # The original one-second spacing expires while the response body is still pending.
        await asyncio.sleep(1.1)
        self.assertEqual(len(self.calls), 1)
        self.assertFalse(second.done())
        for task in (first, second):
            task.cancel()
        results = await asyncio.gather(first, second, return_exceptions=True)
        self.assertTrue(all(isinstance(result, asyncio.CancelledError) for result in results))
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertEqual(self.ledger.snapshot().calls_used, 1)

    async def test_429_body_timeout_keeps_cooldown_and_unknown_reservation(self):
        class SlowStream(httpx.AsyncByteStream):
            async def __aiter__(self):
                await asyncio.Event().wait()
                yield b"{}"

        gateway = self.gateway(
            lambda request: httpx.Response(429, headers={"Retry-After": "60"}, stream=SlowStream()),
            bypass_rate=False,
            timeout_s=0.05,
        )
        await self.fails("timeout", gateway.bind("a", 0, self.budget).complete(self.request()))
        self.assertGreater(gateway._cooldown_until, self.loop.time() + 59)
        await self.fails("timeout", gateway.bind("b", 0, self.budget).complete(self.request()))
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(self.ledger.snapshot().held_units, 720)
        self.assertEqual(self.ledger.snapshot().calls_used, 1)

    async def test_individual_input_and_output_bound_violations_halt(self):
        for bound, cap, prompt, completion in (
            (490, 1, 0, 40),
            (1, 100, 100, 0),
            (120, 10, 120, 40),
        ):
            self.ledger = BudgetLedger(limit_units=20000, deadline_monotonic=self.deadline)
            body = payload(usage={"prompt_tokens": prompt, "completion_tokens": completion})
            gateway = self.gateway(
                lambda request: httpx.Response(200, json=body),
                input_bound=lambda request, body: bound,
            )
            await self.fails(
                "accounting_violation",
                gateway.bind("a", 0, self.budget).complete(self.request(max_output_tokens=cap)),
            )
            self.assertTrue(self.ledger.snapshot().blocked)
            self.assertEqual(self.ledger.snapshot().spent_units, prompt + 10 * completion)
            self.assertEqual(self.ledger.snapshot().held_units, 0)

    async def test_partial_unknown_over_cap_halts_without_inventing_exact_usage(self):
        body = payload(usage={"completion_tokens": 40})
        gateway = self.gateway(
            lambda request: httpx.Response(200, json=body),
            profiles={},
            accounting_mode="best_effort",
            input_bound=lambda request, body: 490,
        )
        await self.fails(
            "accounting_violation",
            gateway.bind("a", 0, self.budget).complete(self.request(max_output_tokens=1)),
        )
        self.assertTrue(self.ledger.snapshot().blocked)
        self.assertEqual(self.ledger.snapshot().held_units, 500)
        self.assertEqual(self.ledger.snapshot().spent_units, 0)

    async def test_custom_output_parameter_and_broken_event_sink(self):
        def broken_sink(event):
            raise RuntimeError("telemetry unavailable")

        def handler(request):
            wire = json.loads(request.content)
            self.assertEqual(wire["max_completion_tokens"], 60)
            self.assertNotIn("max_tokens", wire)
            return httpx.Response(200, json=payload())

        gateway = self.gateway(
            output_token_parameter="max_completion_tokens",
            event_sink=broken_sink,
            transport=httpx.MockTransport(handler),
        )
        reply = await gateway.bind("a", 0, self.budget).complete(self.request())
        self.assertEqual(reply.usage.cost_units, 520)

    async def test_configuration_errors_do_not_include_secrets(self):
        for env in (
            {},
            {**ENV, "LLM_BASE_URL": "http://plain.invalid"},
            {**ENV, "LLM_BASE_URL": "https://user:secret@proxy.invalid"},
            {**ENV, "LLM_BASE_URL": "https://proxy.invalid/?key=secret"},
            {**ENV, "LLM_API_KEY": "bad\nkey"},
            {**ENV, "LLM_API_KEY": "   "},
        ):
            with self.assertRaises(ContractError) as error:
                self.gateway(environ=env)
            self.assertNotIn("secret", str(error.exception))
        for options in (
            {"max_concurrent_requests": 6},
            {"requests_per_minute": 61},
            {"requests_per_minute": True},
            {"timeout_s": 0},
            {"input_bound": None},
        ):
            with self.assertRaises(ContractError):
                self.gateway(**options)

    async def test_real_transport_configuration_disables_retries_and_environment_proxies(self):
        # Construct/close only; network handle_async_request is forbidden by setUp.
        original = httpx.AsyncHTTPTransport
        with (
            patch.dict(os.environ, ENV),
            patch(
                "nsu_budget_agent.llm.gateway.httpx.AsyncHTTPTransport", wraps=original
            ) as factory,
        ):
            gateway = NeuralDeepGateway(budget=self.ledger, input_bound=lambda request, body: 120)
            self.addAsyncCleanup(gateway.aclose)
            self.assertEqual(factory.call_args.kwargs["retries"], 0)
            self.assertFalse(factory.call_args.kwargs["trust_env"])
            await self.fails(
                "unsupported", gateway.bind("a", 0, self.budget).complete(self.request())
            )
            self.assertEqual(self.calls, [])

    async def test_cancellation_from_sink_before_dispatch_releases_reservation(self):
        def cancelled_sink(event):
            raise asyncio.CancelledError

        gateway = self.gateway(event_sink=cancelled_sink)
        with self.assertRaises(asyncio.CancelledError):
            await gateway.bind("a", 0, self.budget).complete(self.request())
        self.assertEqual(self.calls, [])
        self.assertEqual(self.ledger.snapshot().held_units, 0)
        self.assertEqual(self.ledger.snapshot().calls_used, 0)

    async def test_closed_gateway_rejects_new_calls(self):
        gateway = self.gateway()
        session = gateway.bind("a", 0, self.budget)
        await gateway.aclose()
        await gateway.aclose()
        with self.assertRaises(ContractError):
            await session.complete(self.request())


if __name__ == "__main__":
    unittest.main()
