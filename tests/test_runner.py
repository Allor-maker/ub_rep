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
    if "task specification" in system:
        return "### FILE: mylib.py\n```python\n" + FIXED + "```"
    if "IGNORE" in user:
        return "UNSOLVABLE"
    return "```python\nfrom agentlib import rows\nprint(sum(r['units'] for r in rows('sales.csv') if r['region']=='east'))\n```"


def build(root, fake, **overrides):
    overrides.setdefault("verify_types", ())
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
        summary = asyncio.run(build(self.root, fake, verify_top=0.0).run())
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
        summary = asyncio.run(build(self.root, fake, budget=10.0, soft_budget=False).run())
        self.assertEqual(summary["answered"], 1)
        self.assertEqual(len(fake.requests), 0)

    def test_soft_budget_still_tries_deferred_tasks(self):
        # Our count says "no budget", but only the organizers stop the run: keep trying, cheapest first.
        fake = FakeLLM(handler)
        summary = asyncio.run(build(self.root, fake, budget=10.0).run())
        self.assertEqual(summary["answered"], 3)
        self.assertEqual(len(fake.requests), 2)

    def test_emulated_external_stop_loses_the_reply_that_crosses_the_budget(self):
        fake = FakeLLM(handler)
        summary = asyncio.run(build(self.root, fake, budget=10.0, stop_at_budget=True).run())
        self.assertEqual(summary["answered"], 1)  # only the free fast-path answer
        self.assertEqual(len(fake.requests), 1)
        self.assertIn("emulated external stop", summary["llm_disabled_reason"])

    def test_unsolvable_is_accepted_without_second_opinion(self):
        root = temp_root()
        make_task(root, "weird", "IGNORE: who is the CEO of a company not in the files?", task_type="search")
        fake = FakeLLM(handler)
        summary = asyncio.run(build(root, fake).run())
        self.assertEqual(summary["infeasible"], 1)
        self.assertEqual([r["model"] for r in fake.requests], ["qwen3.8-27b-noreason"])
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
        make_task(root, "coding_bad", CODING, task_type="coding", check="unittest", reward=1,
                  files={"mylib.py": BUGGY, "test_mylib.py": contradictory})
        make_task(root, "valuable", "A warehouse starts with 10 boxes. It ships 2 boxes per day for 3 days. "
                  "How many boxes does it have at the end?", reward=9)  # makes coding_bad low-value
        fake = FakeLLM(lambda m, model: "### FILE: mylib.py\n```python\n" + FIXED + "```")
        summary = asyncio.run(build(root, fake).run())
        self.assertEqual(summary["failed"], 1)
        self.assertEqual(len(fake.requests), 2)  # one repair, "no progress" stop; low value: no second pass
        self.assertEqual((root / "coding_bad" / "workspace" / "mylib.py").read_text(), BUGGY)

    def test_high_reward_answer_is_verified_and_replaced_on_disagreement(self):
        def two_models(messages, model):
            if "task specification" in messages[0]["content"]:
                return "### FILE: mylib.py\n```python\n" + FIXED + "```"
            value = 14 if model == "qwen3.8-27b" else 13  # the cheap model is wrong
            return f"```python\nprint({value})\n```"

        def two_models_judged(messages, model):
            if "Two Python programs" in messages[0]["content"]:
                return "B"
            return two_models(messages, model)

        fake = FakeLLM(two_models_judged)
        summary = asyncio.run(build(self.root, fake, verify_top=0.25).run())  # data_llm has the top reward (5)
        self.assertEqual((self.root / "data_llm" / "answer.txt").read_text(), "14\n")
        self.assertEqual([r["model"] for r in fake.requests].count("qwen3.8-27b"), 2)  # verifier + arbiter, top only
        self.assertEqual(summary["answered"], 3)

    def test_unsolvable_gets_no_second_opinion_whatever_the_reward(self):
        root = temp_root()
        make_task(root, "weird_low", "IGNORE: who is the CEO?", task_type="search", reward=1)
        make_task(root, "weird_high", "IGNORE: who is the CFO?", task_type="search", reward=50)
        fake = FakeLLM(handler)
        asyncio.run(build(root, fake).run())
        models = [r["model"] for r in fake.requests]
        self.assertEqual(models.count("qwen3.8-27b-noreason"), 2)
        self.assertEqual(models.count("qwen3.8-27b"), 0)

    def test_precheck_skips_low_reward_question_about_unknown_person(self):
        root = temp_root()
        kb = {f"kb/p{i}.md": f"# Person {chr(65 + i)}x\n\nBorn: 19{50 + i}\n" for i in range(5)}
        make_task(root, "ghost_low", "Using the profiles in workspace/kb/, tell me Olga Petrova's birth year.",
                  task_type="search", reward=1, files=kb)
        make_task(root, "ghost_high", "Using the profiles in workspace/kb/, tell me Ivan Petrov's birth year.",
                  task_type="search", reward=90, files=kb)
        make_task(root, "filler", "x", task_type="math", reward=2)
        make_task(root, "filler2", "y", task_type="math", reward=3)
        fake = FakeLLM(lambda m, model: "UNSOLVABLE")
        asyncio.run(build(root, fake, second_pass=False).run())
        asked = " ".join(r["messages"][1]["content"] for r in fake.requests)
        self.assertNotIn("Olga Petrova", asked)  # rejected for free
        self.assertIn("Ivan Petrov", asked)  # high reward: the model gets a look

    def test_coding_spec_review_and_implement_modes(self):
        root = temp_root()
        stub = "def double(x):\n    return None\n"
        weak = ("import unittest\nfrom mylib import double\n\nclass T(unittest.TestCase):\n"
                "    def test_none(self):\n        self.assertIsNone(double(None))\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n")
        spec = ("Implement `double(x)`: return 2*x, None for None. Hidden tests apply. "
                "Run `python3 test_mylib.py`. Write ONLY the word DONE to answer.txt.")
        make_task(root, "review", spec, task_type="coding", check="unittest",
                  files={"mylib.py": stub, "test_mylib.py": weak})
        make_task(root, "implement", "Implement `double(x)` returning 2*x. Write ONLY the word DONE to answer.txt.",
                  task_type="coding", check="unittest", files={"mylib.py": stub})
        good = "def double(x):\n    return None if x is None else 2 * x\n"
        fake = FakeLLM(lambda m, model: "### FILE: mylib.py\n```python\n" + good + "```")
        asyncio.run(build(root, fake, second_pass=False).run())
        self.assertEqual(len(fake.requests), 2)  # weak passing tests did not short-circuit; no tests != pass
        self.assertEqual({r["model"] for r in fake.requests}, {"qwen3.8-27b"})  # spec tasks -> reasoning model
        for task_id in ("review", "implement"):
            self.assertEqual((root / task_id / "workspace" / "mylib.py").read_text(), good)
            self.assertEqual((root / task_id / "answer.txt").read_text(), "DONE\n")
        self.assertIn("visible tests already pass", fake.requests[0]["messages"][1]["content"]
                      + fake.requests[1]["messages"][1]["content"])

    def test_math_llm_answers_are_verified_by_type(self):
        root = temp_root()
        make_task(root, "m", "Compute (2+3)*4 in some unusual way.", task_type="math", reward=1)
        make_task(root, "d", "How many units were sold in the east region? Read sales.csv.",
                  task_type="data_analysis", reward=1, files={"sales.csv": SALES})
        def answer(messages, model):
            if "Two Python programs" in messages[0]["content"]:
                return "B"
            return f"```python\nprint({20 if model == 'qwen3.8-27b' else 25})\n```"

        fake = FakeLLM(answer)
        asyncio.run(build(root, fake, verify_types=("math",)).run())
        self.assertEqual((root / "m" / "answer.txt").read_text(), "20\n")  # verifier + judge fixed the maths
        self.assertEqual([r["model"] for r in fake.requests].count("qwen3.8-27b"), 2)  # data not verified

    def test_verifier_alone_cannot_overturn_an_answer(self):
        root = temp_root()
        make_task(root, "m", "Compute (2+3)*4 in some unusual way.", task_type="math", reward=1)

        def flaky(messages, model):
            if "Two Python programs" in messages[0]["content"]:
                return "A"  # the judge sees the verifier's bug
            if model == "qwen3.8-27b-noreason":
                return "```python\nprint(20)\n```"
            return "```python\nprint(19)\n```"  # systematic verifier mistake

        asyncio.run(build(root, FakeLLM(flaky), verify_types=("math",)).run())
        self.assertEqual((root / "m" / "answer.txt").read_text(), "20\n")

    def test_summary_is_json_serializable(self):
        summary = asyncio.run(build(self.root, FakeLLM(handler)).run())
        json.dumps(summary)


if __name__ == "__main__":
    unittest.main()
