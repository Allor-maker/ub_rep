"""One cumulative 500000-Unit allowance across fresh experiment processes."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx
from tests.benchmarks import a75


class SessionTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(dir=Path.cwd())
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for replacement in (
            patch.object(a75, "JOURNAL", self.root / "journal.json"),
            patch.dict(
                os.environ,
                {"A75_MODE": "live", "A75_RUN": str(self.root), "LLM_API_KEY": "UNIT-TEST-SECRET"},
            ),
        ):
            replacement.start()
            self.addCleanup(replacement.stop)
        a75.save(
            a75.JOURNAL,
            {
                "spent_units": 0,
                "held_units": 0,
                "http": 0,
                "closed": False,
                "calls": [],
                "quota_can_request": True,
            },
        )
        self.dispatched = 0

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
                        "content": json.dumps({"role": "solve", "instruction": "public test"}),
                    }
                ],
            },
        )

    async def transport(self, *, usage=True, status=200, timeout=False):
        transport = a75.SessionTransport()
        await transport.inner.aclose()

        def handler(request):
            self.dispatched += 1
            self.assertGreater(a75.journal()["held_units"], 0)
            if timeout:
                raise httpx.ReadTimeout("unknown outcome")
            body = {"choices": [{"message": {"content": "UNIT-TEST-SECRET"}}]}
            if usage:
                body["usage"] = (
                    usage
                    if isinstance(usage, dict)
                    else {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}
                )
            return httpx.Response(status, json=body)

        transport.inner = httpx.MockTransport(handler)
        self.addAsyncCleanup(transport.aclose)
        return transport

    async def test_spent_accumulates_and_content_redaction(self):
        for _ in range(2):
            await (await self.transport()).handle_async_request(self.request())
        self.assertEqual(a75.journal()["spent_units"], 600)
        self.assertEqual(a75.journal()["held_units"], 0)
        self.assertNotIn("UNIT-TEST-SECRET", (self.root / "model.jsonl").read_text())

    async def test_next_reservation_cannot_exceed_total_allowance(self):
        state = a75.journal()
        state["spent_units"] = 499999
        a75.save(a75.JOURNAL, state)
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
        self.assertEqual(self.dispatched, 0)

    async def test_unknown_usage_keeps_hold_and_blocks_next_experiment(self):
        await (await self.transport(usage=False)).handle_async_request(self.request())
        self.assertGreater(a75.journal()["held_units"], 0)
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
        self.assertEqual(self.dispatched, 1)

    async def test_error_usage_is_not_treated_as_free(self):
        await (await self.transport(status=429)).handle_async_request(self.request())
        self.assertGreater(a75.journal()["held_units"], 0)
        self.assertTrue(a75.journal()["closed"])
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
        self.assertEqual(self.dispatched, 1)

    async def test_timeout_survives_transport_reconstruction(self):
        with self.assertRaises(httpx.ReadTimeout):
            await (await self.transport(timeout=True)).handle_async_request(self.request())
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
        self.assertGreater(a75.journal()["held_units"], 0)
        self.assertEqual(self.dispatched, 1)

    async def test_reported_cap_breach_closes_even_if_cost_fits_reserve(self):
        await (
            await self.transport(
                usage={"prompt_tokens": 100, "completion_tokens": 65, "total_tokens": 165}
            )
        ).handle_async_request(self.request())
        self.assertEqual(a75.journal()["spent_units"], 750)
        self.assertTrue(a75.journal()["closed"])
        with self.assertRaises(RuntimeError):
            await (await self.transport()).handle_async_request(self.request())
