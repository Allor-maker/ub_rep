import asyncio
import json
import os
import stat
import unittest
from pathlib import Path

from helpers import make_task, temp_root
from nsu_budget_agent.config import parse_config
from nsu_budget_agent.context import build_preview
from nsu_budget_agent.publish import Publisher
from nsu_budget_agent.sandbox import make_sandbox, remove_sandbox, run_python
from nsu_budget_agent.solvers import fastpath
from nsu_budget_agent.solvers.coding import answer_word, is_test_file, test_command
from nsu_budget_agent.solvers.common import extract_code, normalize_answer
from nsu_budget_agent.tasks import load_tasks

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the "
        "file answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
KB = {
    "kb/person_irina_smith.md": "# Irina Smith\n\nBorn: 1983\nEmployer: Dune Robotics\n",
    "kb/person_sergei_garcia.md": "# Sergei Garcia\n\nBorn: 1985\nEmployer: Fjord Data\nMentor: Irina Smith\n",
    "kb/company_dune_robotics.md": "# Dune Robotics\n\nHeadquarters: Osaka\nFounded: 2001\n",
    "kb/company_fjord_data.md": "# Fjord Data\n\nHeadquarters: Leeds\nFounded: 2016\n",
}
SALES = "order_id,region,product,month,units,price\n1,west,widget,7,5,3\n2,east,widget,8,10,3\n3,east,gizmo,7,4,1\n4,west,widget,9,100,1\n"


def load_one(root):
    return load_tasks(root)[0]


class ConfigTests(unittest.TestCase):
    def test_official_launch_needs_nothing_but_tasks_dir(self):
        config = parse_config(["--tasks-dir", "/data/tasks", "--some-new-flag", "x"], environ={})
        self.assertEqual(config.tasks_dir, Path("/data/tasks"))
        self.assertIsNone(config.budget)
        self.assertFalse(config.llm_enabled)

    def test_env_overrides(self):
        env = {"TASKS": "/t", "OUT": "/o", "BUDGET": "5000", "LLM_BASE_URL": "http://x/v1/", "LLM_API_KEY": "k",
               "AGENT_MODEL": "qwen3.6-35b-a3b-noreason", "AGENT_SECOND_MODEL": "none"}
        config = parse_config([], environ=env)
        self.assertEqual((config.tasks_dir, config.out, config.budget), (Path("/t"), Path("/o"), 5000.0))
        self.assertEqual(config.base_url, "http://x/v1")
        self.assertEqual(config.model, "qwen3.6-35b-a3b-noreason")
        self.assertIsNone(config.second_model)

    def test_invalid_model_falls_back_to_default(self):
        config = parse_config([], environ={"AGENT_MODEL": "gpt-5"})
        self.assertEqual(config.model, "qwen3.8-27b-noreason")


class LoaderTests(unittest.TestCase):
    def test_expected_never_loaded_and_broken_task_skipped(self):
        root = temp_root()
        make_task(root, "math_1", "A warehouse starts with 5 boxes. How many boxes does it have at the end?")
        (root / "broken").mkdir()
        (root / "broken" / "task.json").write_text("{not json")
        tasks = load_tasks(root)
        self.assertEqual([t.id for t in tasks], ["math_1"])
        self.assertNotIn("expected", tasks[0].meta)
        self.assertNotIn("SECRET", json.dumps(tasks[0].meta))


class FastPathTests(unittest.TestCase):
    def test_inventory_template(self):
        root = temp_root()
        text = ("A warehouse starts with 1752 boxes. Then the stock is split equally into 2 warehouses and it keeps "
                "one share (rounded down). Then the stock is doubled by merging with a sister warehouse. Then the "
                "stock is split equally into 3 warehouses and it keeps one share (rounded down). Then the stock is "
                "doubled by merging with a sister warehouse. Then the stock is split equally into 3 warehouses and "
                "it keeps one share (rounded down). How many boxes does it have at the end? Compute it with the "
                "bash tool (python3 -c)." + TAIL)
        make_task(root, "m", text)
        self.assertEqual(fastpath.solve(load_one(root)).answer, "389")

    def test_shipping_template(self):
        root = temp_root()
        make_task(root, "m", "A warehouse starts with 1669 boxes. It ships 36 boxes per day for 3 days. It receives "
                  "2 deliveries of 87 boxes each. How many boxes does it have at the end?" + TAIL)
        self.assertEqual(fastpath.solve(load_one(root)).answer, "1735")

    def test_unknown_sentence_goes_to_llm(self):
        root = temp_root()
        make_task(root, "m", "A warehouse starts with 100 boxes. It loses 10 percent to damage. "
                  "How many boxes does it have at the end?" + TAIL)
        self.assertIsNone(fastpath.solve(load_one(root)))

    def test_csv_units(self):
        root = temp_root()
        make_task(root, "d", "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). "
                  "How many units of product 'widget' were sold in months 7 through 8 inclusive? Use bash with "
                  "python3 to compute it." + TAIL, task_type="data_analysis", files={"sales.csv": SALES})
        self.assertEqual(fastpath.solve(load_one(root)).answer, "15")

    def test_csv_extra_condition_goes_to_llm(self):
        root = temp_root()
        make_task(root, "d", "The file sales.csv has columns order_id,region,product,month,units,price (4 rows). "
                  "How many units of product 'widget' were sold in months 7 through 8 inclusive? Only count the "
                  "west region." + TAIL, task_type="data_analysis", files={"sales.csv": SALES})
        self.assertIsNone(fastpath.solve(load_one(root)))

    def test_csv_missing_column_is_infeasible(self):
        root = temp_root()
        make_task(root, "d", "The file sales.csv has columns order_id,region,product,month,units,price (1 rows). "
                  "How many units of product 'widget' were sold in months 7 through 8 inclusive?" + TAIL,
                  task_type="data_analysis", files={"sales.csv": "order_id,product,month\n1,widget,7\n"})
        self.assertEqual(fastpath.solve(load_one(root)).status, "infeasible")

    def test_kb_born_and_employer_hq(self):
        intro = "The directory kb/ contains a small knowledge base of markdown files. "
        tail = " Use the file tools or bash (ls, grep, cat) to find it." + TAIL
        root = temp_root()
        make_task(root, "s1", intro + "In which year was Irina Smith born?" + tail, task_type="search", files=KB)
        self.assertEqual(fastpath.solve(load_one(root)).answer, "1983")
        root = temp_root()
        make_task(root, "s2", intro + "In which city is the headquarters of Sergei Garcia's employer?" + tail,
                  task_type="search", files=KB)
        self.assertEqual(fastpath.solve(load_one(root)).answer, "Leeds")
        root = temp_root()
        make_task(root, "s3", intro + "In which year was Olga Petrova born?" + tail, task_type="search", files=KB)
        result = fastpath.solve(load_one(root))
        self.assertEqual(result.status, "infeasible")


class PublishTests(unittest.TestCase):
    def test_writes_next_to_task_json_and_out(self):
        root = temp_root()
        make_task(root, "d", "x", files={"sales.csv": SALES})
        make_task(root, "m", "y")
        out_dir = root / "out"
        out_dir.mkdir()
        publisher = Publisher(out_dir)
        tasks = {t.id: t for t in load_tasks(root)}
        publisher.publish(tasks["d"], "158")
        publisher.publish(tasks["m"], "42")
        self.assertEqual((root / "d" / "answer.txt").read_text(), "158\n")
        self.assertFalse((root / "d" / "workspace" / "answer.txt").exists())
        self.assertFalse((root / "m" / "workspace").exists())
        mode = stat.S_IMODE(os.stat(root / "m" / "answer.txt").st_mode)
        self.assertEqual(mode, 0o644)
        self.assertEqual(json.loads((out_dir / "answers.json").read_text()), {"d": "158", "m": "42"})

    def test_out_as_file(self):
        root = temp_root()
        make_task(root, "m", "y")
        publisher = Publisher(root / "res" / "answers.json")
        publisher.publish(load_one(root), "7")
        self.assertEqual(json.loads((root / "res" / "answers.json").read_text()), {"m": "7"})


class SandboxTests(unittest.TestCase):
    def test_secret_not_visible_and_inputs_untouched(self):
        root = temp_root()
        task_dir = make_task(root, "d", "x", files={"sales.csv": SALES})
        os.environ["LLM_API_KEY"] = "super-secret"
        try:
            box = make_sandbox(task_dir, task_dir / "workspace")
            code = ("import os\nprint(os.environ.get('LLM_API_KEY'))\nopen('sales.csv','w').write('x')\n"
                    "print(open('workspace/sales.csv').read())")
            result = asyncio.run(run_python(code, box, timeout=10))
            remove_sandbox(box)
        finally:
            del os.environ["LLM_API_KEY"]
        self.assertTrue(result.ok, result.stderr)
        self.assertNotIn("super-secret", result.stdout)
        self.assertEqual((task_dir / "workspace" / "sales.csv").read_text(), SALES)

    def test_timeout_and_agentlib(self):
        root = temp_root()
        task_dir = make_task(root, "s", "x", files=KB)
        box = make_sandbox(task_dir, task_dir / "workspace")
        slow = asyncio.run(run_python("import time\ntime.sleep(30)", box, timeout=1))
        self.assertTrue(slow.timed_out)
        lib = asyncio.run(run_python(
            "from agentlib import kb\nk=kb('kb')\nprint(k[k['sergei garcia']['Employer']]['HEADQUARTERS'])", box,
            timeout=10))
        remove_sandbox(box)
        self.assertEqual(lib.stdout.strip(), "Leeds", lib.stderr)


class HelperTests(unittest.TestCase):
    def test_normalize_answer(self):
        self.assertEqual(normalize_answer(" 158.0 "), "158")
        self.assertEqual(normalize_answer("'Osaka'"), "Osaka")
        self.assertEqual(normalize_answer("Answer: 42"), "42")
        self.assertEqual(normalize_answer("Osaka."), "Osaka")
        self.assertEqual(normalize_answer("3.25"), "3.25")

    def test_extract_code(self):
        self.assertEqual(extract_code("text\n```python\nprint(1)\n```\n"), "print(1)\n")
        self.assertIsNone(extract_code("UNSOLVABLE"))

    def test_coding_helpers(self):
        self.assertTrue(is_test_file("test_mylib.py"))
        self.assertTrue(is_test_file("tests/helpers.py"))
        self.assertFalse(is_test_file("mylib.py"))
        self.assertEqual(answer_word("write ONLY the word DONE to the file"), "DONE")
        root = temp_root()
        make_task(root, "c", "Run `python3 test_mylib.py` in the current directory; fix it.", task_type="coding")
        argv = test_command(load_one(root), root)
        self.assertEqual(argv[1:], ["test_mylib.py"])
        make_task(root, "c2", "Run `python3 x.py; rm -rf /` now.", task_type="coding")
        evil = [t for t in load_tasks(root) if t.id == "c2"][0]
        self.assertIn("unittest", test_command(evil, root))

    def test_preview_lists_kb_keys_and_csv_values(self):
        root = temp_root()
        task_dir = make_task(root, "s", "x", files={**KB, "kb/person_a.md": "# A\n2019 summit host city: Graz\n",
                                                   "kb/x.md": "# X\n", "kb/y.md": "# Y\n", "sales.csv": SALES})
        preview = build_preview(task_dir / "workspace")
        self.assertIn("2019 summit host city", preview)
        self.assertIn("product={gizmo, widget}", preview)


if __name__ == "__main__":
    unittest.main()
