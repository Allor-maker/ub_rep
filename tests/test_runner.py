"""End-to-end runs with a fake LLM endpoint (no network)."""

import asyncio
import json
import unittest

from helpers import FakeLLM, make_task, temp_root
from nsu_budget_agent.config import Config
from nsu_budget_agent.runner import Runner

BUGGY = "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n"
FIXED = "def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n"
TESTS = ("import unittest\nimport mylib as m\n\nclass T(unittest.TestCase):\n    def test_total(self):\n"
         "        self.assertEqual(m.total([1, 2, 3]), 6)\n\nif __name__ == '__main__':\n    unittest.main()\n")
CODING = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not "
          "edit the tests) until all tests pass. When you are done, write ONLY the word DONE to the file answer.txt.")


def handler(messages, model):
    system, user = messages[0]["content"], messages[1]["content"]
    assert "SECRET-EXPECTED" not in user
    if "fix bugs" in system:
        return "### FILE: mylib.py\n```python\n" + FIXED + "```"
    if "IGNORE" in user:
        return "UNSOLVABLE"
    return "```python\nfrom agentlib import rows\nprint(sum(r['units'] for r in rows('sales.csv') if r['region']=='east'))\n```"


def build(root, fake, **overrides):
    config = Config(tasks_dir=root, base_url="http://fake/v1", api_key="k", concurrency=2, deadline_s=120,
                    log_dir=root / "logs", **overrides)
    return Runner(config, transport=fake.transport())


SALES = "order_id,region,product,month,units,price\n1,west,widget,7,5,3\n2,east,widget,8,10,3\n3,east,gizmo,7,4,1\n"


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.root = temp_root()
        make_task(self.root, "math_free", "A warehouse starts with 10 boxes. It ships 2 boxes per day for 3 days. "
                  "How many boxes does it have at the end?", reward=1)
        make_task(self.root, "data_llm", "How many units were sold in the east region? Read sales.csv.",
                  task_type="data_analysis", reward=5, files={"sales.csv": SALES})
        make_task(self.root, "coding_llm", CODING, task_type="coding", check="unittest", reward=3,
                  files={"mylib.py": BUGGY, "test_mylib.py": TESTS})

    def test_full_run(self):
        fake = FakeLLM(handler)
        summary = asyncio.run(build(self.root, fake).run())
        self.assertEqual((self.root / "math_free" / "answer.txt").read_text(), "4\n")
        self.assertEqual((self.root / "data_llm" / "answer.txt").read_text(), "14\n")
        self.assertEqual((self.root / "coding_llm" / "answer.txt").read_text(), "DONE\n")
        self.assertEqual((self.root / "coding_llm" / "workspace" / "mylib.py").read_text(), FIXED)
        self.assertEqual((self.root / "coding_llm" / "workspace" / "test_mylib.py").read_text(), TESTS)
        self.assertEqual(summary["answered"], 3)
        self.assertEqual(summary["answered_free"], 1)
        self.assertEqual(summary["llm_calls"], 2)
        self.assertAlmostEqual(summary["tokens_spent"], 2 * (1000 / 10 + 100))
        lines = (self.root / "logs" / "trajectories.jsonl").read_text().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertTrue(all(m["model"] == "qwen3.8-27b-noreason" for m in fake.requests))

    def test_budget_exhausted_proxy_keeps_free_answers(self):
        fake = FakeLLM(handler, status=402)
        summary = asyncio.run(build(self.root, fake).run())
        self.assertEqual((self.root / "math_free" / "answer.txt").read_text(), "4\n")
        self.assertFalse((self.root / "data_llm" / "answer.txt").exists())
        self.assertEqual(summary["answered"], 1)
        self.assertIn("402", summary["llm_disabled_reason"])
        self.assertLessEqual(len(fake.requests), 2)

    def test_known_budget_skips_unaffordable_tasks(self):
        fake = FakeLLM(handler)
        summary = asyncio.run(build(self.root, fake, budget=10.0).run())
        self.assertEqual(summary["answered"], 1)
        self.assertEqual(len(fake.requests), 0)

    def test_unsolvable_gets_second_opinion_then_stops(self):
        root = temp_root()
        make_task(root, "weird", "IGNORE: who is the CEO of a company not in the files?", task_type="search")
        fake = FakeLLM(handler)
        summary = asyncio.run(build(root, fake).run())
        self.assertEqual(summary["infeasible"], 1)
        self.assertEqual([r["model"] for r in fake.requests], ["qwen3.8-27b-noreason", "qwen3.8-27b"])
        self.assertFalse((root / "weird" / "answer.txt").exists())

    def test_failed_coding_leaves_workspace_untouched(self):
        fake = FakeLLM(lambda m, model: "### FILE: test_mylib.py\n```python\nprint('hacked')\n```")
        asyncio.run(build(self.root, fake, second_model=None, second_pass=False).run())
        self.assertEqual((self.root / "coding_llm" / "workspace" / "mylib.py").read_text(), BUGGY)
        self.assertEqual((self.root / "coding_llm" / "workspace" / "test_mylib.py").read_text(), TESTS)
        self.assertFalse((self.root / "coding_llm" / "answer.txt").exists())

    def test_contradictory_tests_stop_without_second_pass(self):
        root = temp_root()
        contradictory = TESTS.replace(
            "    def test_total(self):",
            "    def test_total_again(self):\n        self.assertNotEqual(m.total([1, 2, 3]), 6)\n\n    def test_total(self):",
        )
        make_task(root, "coding_bad", CODING, task_type="coding", check="unittest",
                  files={"mylib.py": BUGGY, "test_mylib.py": contradictory})
        fake = FakeLLM(lambda m, model: "### FILE: mylib.py\n```python\n" + FIXED + "```")
        summary = asyncio.run(build(root, fake).run())
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(fake.requests), 2)  # one repair, then "no progress" stop; no second pass
        self.assertEqual((root / "coding_bad" / "workspace" / "mylib.py").read_text(), BUGGY)

    def test_summary_is_json_serializable(self):
        summary = asyncio.run(build(self.root, FakeLLM(handler)).run())
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
