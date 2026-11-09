"""Generate a practice set in the organizers' task.json format, including infeasible tasks.

Families mirror the public examples (exact phrasing) plus paraphrased/extended variants,
so we measure both the zero-token fast path and LLM generalization. About 20% of tasks
are infeasible (as in the final set). Rewards are random, unrelated to difficulty.

The answer key is written OUTSIDE the tasks dir: python tools/gen_tasks.py OUT_DIR --n 60
-> OUT_DIR/tasks/<id>/task.json (+workspace), OUT_DIR/answers.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

TAIL = (" Work only inside the current directory. When you are done, write ONLY the final answer to the file "
        "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
TAIL_CODE = (" Work only inside the current directory. When you are done, write ONLY the word DONE to the file "
             "answer.txt in the current directory (for example with bash: echo 42 > answer.txt).")
FIRST = ["Alice", "Boris", "Chen", "Dmitri", "Farid", "Grace", "Hiro", "Irina", "Jonas", "Kira", "Maya", "Nikolai",
         "Olga", "Pavel", "Quinn", "Rosa", "Sergei", "Tanya"]
LAST = ["Smith", "Petrov", "Sato", "Haddad", "Wang", "Novak", "Garcia", "Kowalski", "Muller", "Ivanova"]
COMPANIES = ["Arktika Labs", "Juniper Foods", "Fjord Data", "Bluefin Systems", "Iris Optics", "Ember Bio",
             "Granite AI", "Cobalt Works", "Dune Robotics", "Harbor Dynamics"]
CITIES = ["Osaka", "Leeds", "Graz", "Porto", "Brno", "Ghent", "Turin", "Riga", "Lyon", "Bergen", "Krakow", "Malmo"]
PRODUCTS = ["widget", "gadget", "sprocket", "gizmo"]
REGIONS = ["north", "south", "east", "west"]


def slug(name: str) -> str:
    return name.lower().replace(" ", "_")


# ---------------------------------------------------------------------------- math
def gen_math(rng: random.Random, variant: int):
    n = rng.randint(800, 4000)
    value, parts = n, [f"A warehouse starts with {n} boxes."]
    if variant == 0:  # official chain template
        for _ in range(rng.randint(3, 5)):
            if rng.random() < 0.5:
                k = rng.randint(2, 4)
                parts.append(f"Then the stock is split equally into {k} warehouses and it keeps one share (rounded down).")
                value //= k
            else:
                parts.append("Then the stock is doubled by merging with a sister warehouse.")
                value *= 2
        q = "How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
    elif variant == 1:  # official shipping template
        a, d, r, b = rng.randint(10, 60), rng.randint(2, 9), rng.randint(1, 5), rng.randint(20, 120)
        parts += [f"It ships {a} boxes per day for {d} days.", f"It receives {r} deliveries of {b} boxes each."]
        value = n - a * d + r * b
        q = "How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
    elif variant == 2:  # paraphrase (LLM)
        a, d, pct = rng.randint(10, 60), rng.randint(2, 9), rng.choice([10, 20, 25, 50])
        parts = [f"A depot holds {n} crates.", f"Every day for {d} days it sends out {a} crates.",
                 f"Then {pct} percent of the remaining crates are found damaged and thrown away (round the number "
                 "thrown away down)."]
        rest = n - a * d
        value = rest - rest * pct // 100
        q = "How many crates are left? Compute it with python3."
    else:  # infeasible: initial amount not given
        a, d = rng.randint(10, 60), rng.randint(2, 9)
        parts = ["A warehouse starts with an unknown number of boxes.", f"It ships {a} boxes per day for {d} days."]
        value = None
        q = "How many boxes does it have at the end? Compute it with the bash tool (python3 -c)."
    return " ".join(parts + [q]) + TAIL, {}, None if value is None else str(value), "exact_int"


# ---------------------------------------------------------------------------- data
def gen_data(rng: random.Random, variant: int):
    rows = rng.choice([20, 60, 300])
    header = ["order_id", "region", "product", "month", "units", "price"]
    data = [[str(i + 1), rng.choice(REGIONS), rng.choice(PRODUCTS), str(rng.randint(1, 12)), str(rng.randint(1, 40)),
             str(rng.randint(5, 99))] for i in range(rows)]
    product, a = rng.choice(PRODUCTS), rng.randint(1, 10)
    b = rng.randint(a, 12)
    intro = f"The file sales.csv has columns {','.join(header)} ({rows} rows). "
    if variant == 0:
        q = f"How many units of product '{product}' were sold in months {a} through {b} inclusive?"
        value = sum(int(r[4]) for r in data if r[2] == product and a <= int(r[3]) <= b)
    elif variant == 1:
        region = rng.choice(REGIONS)
        q = f"What was the total revenue (units times price) of product '{product}' in region '{region}'?"
        value = sum(int(r[4]) * int(r[5]) for r in data if r[2] == product and r[1] == region)
    elif variant == 2:
        q = f"How many orders (rows) of product '{product}' have more than 20 units?"
        value = sum(1 for r in data if r[2] == product and int(r[4]) > 20)
    else:  # infeasible: asked column does not exist
        header = ["order_id", "region", "product", "month", "price"]
        data = [r[:4] + r[5:] for r in data]
        q = f"How many units of product '{product}' were sold in months {a} through {b} inclusive?"
        value = None
    csv_text = "\n".join(",".join(r) for r in [header] + data) + "\n"
    text = intro + q + " Use bash with python3 to compute it." + TAIL
    return text, {"sales.csv": csv_text}, None if value is None else str(value), "exact_int"


# ---------------------------------------------------------------------------- search
def gen_search(rng: random.Random, variant: int):
    names = rng.sample([f"{f} {l_}" for f in FIRST for l_ in LAST], 28)
    companies = rng.sample(COMPANIES, 8)
    files, people, comp = {}, {}, {}
    for c in companies:
        comp[c] = {"Headquarters": rng.choice(CITIES), "Founded": str(rng.randint(1950, 2020))}
        files[f"kb/company_{slug(c)}.md"] = f"# {c}\n\nHeadquarters: {comp[c]['Headquarters']}\nFounded: {comp[c]['Founded']}\n"
    for name in names:
        people[name] = {"Born": str(rng.randint(1950, 2002)), "Employer": rng.choice(companies)}
    for name in names:
        people[name]["Mentor"] = rng.choice([n for n in names if n != name])
        p = people[name]
        files[f"kb/person_{slug(name)}.md"] = (f"# {name}\n\nBorn: {p['Born']}\nEmployer: {p['Employer']}\n"
                                               f"Mentor: {p['Mentor']}\n")
    target = rng.choice(names)
    intro = "The directory kb/ contains a small knowledge base of markdown files. "
    tail = " Use the file tools or bash (ls, grep, cat) to find it."
    if variant == 0:
        q, value = f"In which year was {target} born?", people[target]["Born"]
    elif variant == 1:
        q, value = (f"In which city is the headquarters of {target}'s employer?",
                    comp[people[target]["Employer"]]["Headquarters"])
    elif variant == 2:
        mentor = people[target]["Mentor"]
        q, value = (f"In which year was the employer of {target}'s mentor founded?",
                    comp[people[mentor]["Employer"]]["Founded"])
    else:  # infeasible: person absent from the knowledge base
        missing = rng.choice([n for n in (f"{f} {l_}" for f in FIRST for l_ in LAST) if n not in people])
        q, value = f"In which year was {missing} born?", None
    return intro + q + tail + TAIL, files, value, "exact_str"


# ---------------------------------------------------------------------------- coding
FUNCS = {
    "reverse_words": ("def reverse_words(s):\n    return ' '.join(reversed(s.split()))\n",
                      "def reverse_words(s):\n    return ' '.join(sorted(s.split()))\n",
                      "self.assertEqual(m.reverse_words('c a b'), 'b a c')"),
    "total": ("def total(xs):\n    s = 0\n    for x in xs:\n        s += x\n    return s\n",
              "def total(xs):\n    s = 0\n    for x in xs:\n        s -= x\n    return s\n",
              "self.assertEqual(m.total([1, 2, 3]), 6)"),
    "maximum": ("def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x > best:\n            best = x\n    return best\n",
                "def maximum(xs):\n    best = xs[0]\n    for x in xs:\n        if x < best:\n            best = x\n    return best\n",
                "self.assertEqual(m.maximum([3, 9, 2]), 9)"),
    "count_even": ("def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 0)\n",
                   "def count_even(xs):\n    return sum(1 for x in xs if x % 2 == 1)\n",
                   "self.assertEqual(m.count_even([1, 2, 4, 6, 7]), 3)"),
    "first_n": ("def first_n(xs, n):\n    return xs[:n]\n", "def first_n(xs, n):\n    return xs[:n - 1]\n",
                "self.assertEqual(m.first_n([5, 6, 7, 8], 3), [5, 6, 7])"),
    "is_palindrome": ("def is_palindrome(s):\n    t = s.lower()\n    return t == t[::-1]\n",
                      "def is_palindrome(s):\n    t = s.lower()\n    return t == t[1:][::-1]\n",
                      "self.assertTrue(m.is_palindrome('Level'))\n        self.assertFalse(m.is_palindrome('abca'))"),
    "clamp": ("def clamp(x, lo, hi):\n    return max(lo, min(x, hi))\n",
              "def clamp(x, lo, hi):\n    return min(lo, max(x, hi))\n",
              "self.assertEqual(m.clamp(5, 0, 3), 3)\n        self.assertEqual(m.clamp(-1, 0, 3), 0)"),
}


def gen_coding(rng: random.Random, variant: int):
    chosen = rng.sample(sorted(FUNCS), rng.randint(3, 6))
    buggy = set(rng.sample(chosen, rng.randint(1, min(3, len(chosen)))))
    lib = "\n\n".join(FUNCS[f][1] if f in buggy else FUNCS[f][0] for f in chosen)
    tests = "".join(f"    def test_{f}(self):\n        {FUNCS[f][2]}\n\n" for f in chosen)
    expected = "tests_pass"
    if variant == 3:  # infeasible: contradictory tests
        f = chosen[0]
        tests += f"    def test_{f}_again(self):\n        {FUNCS[f][2].replace('assertEqual', 'assertNotEqual').replace('assertTrue', 'assertFalse')}\n\n"
        expected = None
    test_file = "import unittest\n\nimport mylib as m\n\n\nclass T(unittest.TestCase):\n" + tests + \
        "\nif __name__ == '__main__':\n    unittest.main()\n"
    text = ("Run `python3 test_mylib.py` in the current directory; some tests fail. Fix the bugs in mylib.py (do not "
            "edit the tests) until all tests pass." + TAIL_CODE)
    return text, {"mylib.py": lib + "\n", "test_mylib.py": test_file}, expected, "unittest"


GENERATORS = {"math": gen_math, "data_analysis": gen_data, "search": gen_search, "coding": gen_coding}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--n", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--infeasible", type=float, default=0.2)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    tasks_dir = args.out / "tasks"
    tasks_dir.mkdir(parents=True, exist_ok=False)
    answers = {}
    kinds = list(GENERATORS)
    for i in range(args.n):
        kind = kinds[i % len(kinds)]
        variant = 3 if rng.random() < args.infeasible else rng.choice([0, 0, 1, 1, 2])
        text, files, expected, check = GENERATORS[kind](rng, variant)
        task_id = f"{kind}_{9000 + i:04d}"
        task_dir = tasks_dir / task_id
        (task_dir / "workspace").mkdir(parents=True) if files else task_dir.mkdir(parents=True)
        for rel, content in files.items():
            path = task_dir / "workspace" / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        reward = rng.randint(1, 10)
        (task_dir / "task.json").write_text(json.dumps(
            {"task_id": task_id, "task_type": kind, "instruction": text, "reward": reward, "check": check}, indent=2))
        answers[task_id] = {"expected": expected, "reward": reward, "type": kind, "variant": variant,
                            "check": check, "test_file": files.get("test_mylib.py")}
    (args.out / "answers.json").write_text(json.dumps(answers, indent=1))
    print(f"wrote {args.n} tasks to {tasks_dir}")


if __name__ == "__main__":
    main()
