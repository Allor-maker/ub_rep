"""Competition arithmetic interprets reported counters, never provider attestations."""

import unittest

from nsu_budget_agent.budget.config import parse_budget
from nsu_budget_agent.contracts.models import ProviderProfile
from nsu_budget_agent.llm.competition import normalize_competition_usage


class CompetitionUsageTests(unittest.TestCase):
    def usage(self, counts, *, model="qwen3.6-35b-a3b-noreason", reasoning_text=None, success=True):
        message = {"content": "answer"}
        if reasoning_text is not None:
            message["reasoning_content"] = reasoning_text
        return normalize_competition_usage(
            {"usage": counts, "choices": [{"message": message}]},
            request_id="request",
            attempt_id="attempt",
            model=model,
            success=success,
        )

    def test_budget_and_included_reasoning_cache_are_not_discounted_or_counted_twice(self):
        self.assertEqual(parse_budget("100000", unit="competition_tokens"), 1_000_000)
        usage = self.usage(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1100,
                "prompt_tokens_details": {"cached_tokens": 900},
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
            model="qwen3.8-27b",
            reasoning_text="thinking",
        )
        self.assertEqual(usage.status, "exact")
        self.assertEqual(usage.output_tokens, 100)
        self.assertEqual(usage.cost_units, 2000)
        self.assertEqual(usage.reasoning_tokens, 40)
        self.assertFalse(ProviderProfile().input_bound_verified)

    def test_separate_reasoning_is_added_once_when_total_disambiguates(self):
        usage = self.usage(
            {
                "prompt_tokens": 1000,
                "completion_tokens": 100,
                "total_tokens": 1140,
                "reasoning_tokens": 40,
                "completion_tokens_details": {"reasoning_tokens": 40},
            },
            model="qwen3.8-27b",
        )
        self.assertEqual(usage.status, "exact")
        self.assertEqual(usage.output_tokens, 140)
        self.assertEqual(usage.cost_units, 2400)

    def test_missing_ambiguous_or_contradictory_counters_stay_unknown(self):
        cases = (
            None,
            {},
            {"prompt_tokens": 10},
            {"prompt_tokens": 10, "completion_tokens": 20, "reasoning_tokens": 5},
            {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 999},
            {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "reasoning_tokens": 3,
                "completion_tokens_details": {"reasoning_tokens": 4},
            },
            {"prompt_tokens": True, "completion_tokens": 20, "total_tokens": 21},
        )
        for counts in cases:
            with self.subTest(counts=counts):
                self.assertEqual(self.usage(counts).status, "unknown")
        counts = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        self.assertEqual(
            self.usage(counts, model="qwen3.8-27b", reasoning_text="hidden").status, "unknown"
        )
        self.assertEqual(self.usage(counts, success=False).status, "unknown")
        self.assertEqual(self.usage(counts).status, "exact")
