"""Durable cumulative accounting across experiments, with physical HTTP mocked."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from tests.benchmarks import a74


class SessionGuardTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for replacement in (
            patch.object(a74, "JOURNAL", self.root / "journal.json"),
            patch.dict(os.environ, {"A74_RUN": str(self.root), "LLM_API_KEY": "test-secret"}),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        a74.save(
            a74.JOURNAL,
            {"http": 0, "spent_units": 0, "held_units": 0, "quota_ok": True, "calls": []},
        )
        self.requests = []

    def request(self):
        return httpx.Request(
            "POST",
            "https://offline.invalid/v1/chat/completions",
            json={
                "model": "qwen3.6-35b-a3b-noreason",
                "max_tokens": 64,
                "messages": [
                    {
                        "role": "user",
                        "content": json.dumps({"role": "solve", "artifact_provenance": []}),
                    }
                ],
            },
        )

    async def transport(self, *, usage=True, error=False):
        guard = a74.GuardedTransport()
        await guard.inner.aclose()

        def respond(request):
            self.requests.append(request)
            persisted = a74.load_journal()
            self.assertGreater(persisted["held_units"], 0)
            if error:
                raise httpx.ReadTimeout("uncertain dispatch")
            data = {"choices": [{"message": {"content": "test-secret"}}]}
            if usage:
                data["usage"] = (
                    usage
                    if isinstance(usage, dict)
                    else {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
                )
            return httpx.Response(200, json=data)

        guard.inner = httpx.MockTransport(respond)
        self.addAsyncCleanup(guard.aclose)
        return guard

    async def test_exact_usage_accumulates_after_reconstruction(self):
        await (await self.transport()).handle_async_request(self.request())
        await (await self.transport()).handle_async_request(self.request())
        journal = a74.load_journal()
        self.assertEqual(
            (journal["http"], journal["spent_units"], journal["held_units"]), (2, 600, 0)
        )
        self.assertNotIn("test-secret", (self.root / "model.jsonl").read_text())

    async def test_unknown_usage_blocks_new_experiment(self):
        await (await self.transport(usage=False)).handle_async_request(self.request())
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
        self.assertEqual(len(self.requests), 1)
        self.assertGreater(a74.load_journal()["held_units"], 0)

    async def test_timeout_keeps_durable_hold_and_never_retries(self):
        with self.assertRaises(httpx.ReadTimeout):
            await (await self.transport(error=True)).handle_async_request(self.request())
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
        self.assertEqual(len(self.requests), 1)

    async def test_input_or_output_breach_blocks_even_when_total_fits_reserve(self):
        bound = len(self.request().content) + 1024 + 256
        for prompt, completion in ((100, 65), (bound + 1, 0)):
            with self.subTest(prompt=prompt, completion=completion):
                a74.save(
                    a74.JOURNAL,
                    {"http": 0, "spent_units": 0, "held_units": 0, "quota_ok": True, "calls": []},
                )
                before = len(self.requests)
                guard = await self.transport(
                    usage={
                        "prompt_tokens": prompt,
                        "completion_tokens": completion,
                        "total_tokens": prompt + completion,
                    }
                )
                await guard.handle_async_request(self.request())
                journal = a74.load_journal()
                self.assertLess(journal["spent_units"], journal["calls"][0]["reserve"])
                self.assertFalse(journal["quota_ok"])
                with self.assertRaises(RuntimeError):
                    await (await self.transport()).handle_async_request(self.request())
                self.assertEqual(len(self.requests), before + 1)

    async def test_cumulative_budget_includes_next_reservation(self):
        journal = a74.load_journal()
        journal["spent_units"] = 99_000
        a74.save(a74.JOURNAL, journal)
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
        self.assertEqual(self.requests, [])

    async def test_http_count_does_not_reset_between_experiments(self):
        journal = a74.load_journal()
        journal["http"] = 25
        a74.save(a74.JOURNAL, journal)
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
        self.assertEqual(self.requests, [])
