"""Zero-token solvers for exact task templates (the organizers' published phrasing).

Strict by design: EVERY sentence of the instruction must be recognized (apart from
whitelisted boilerplate about tools/answer.txt). One unknown sentence -> not applicable,
the task goes to the LLM. These solvers also detect clear infeasibility for free
(missing file/column/entity/field), so no tokens are spent on such tasks.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass
from pathlib import Path

from ..context import list_files, norm, parse_kb, read_text
from ..tasks import Task


@dataclass
class FastResult:
    status: str  # "answer" | "infeasible"
    answer: str = ""
    reason: str = ""


_SENTENCE_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z])")


def _is_boilerplate(sentence: str) -> bool:
    s = sentence.strip()
    if "answer.txt" in s:
        return True
    if re.match(r"^Work only inside the current (directory|task workspace)\.$", s):
        return True
    if re.match(r"^(Use|Compute it|Compute using|Compute with)\b", s) and re.search(
        r"\b(python3?|bash|file tools|grep)\b", s
    ) and not re.search(r"\d|'", s.replace("python3", "").replace("echo 42", "")):
        return True
    return False


def sentences(instruction: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(instruction.strip()) if s.strip()]


def core_sentences(instruction: str) -> list[str]:
    return [s for s in sentences(instruction) if not _is_boilerplate(s)]


def _resolve(task: Task, rel: str) -> Path | None:
    rel = rel.strip().strip("'\"`").rstrip("/")
    if rel.startswith("workspace/"):
        rel = rel[len("workspace/"):]
    for base in (task.workspace, task.dir):
        candidate = (base / rel).resolve()
        try:
            candidate.relative_to(task.dir.resolve())
        except ValueError:
            continue
        if candidate.exists():
            return candidate
    return None


# ------------------------------------------------------------------- math: inventory

_START = re.compile(r"^A (?:\w+ ){0,2}?(\w+) starts with (\d+) (\w+)\.$", re.I)
_OPS = [
    (re.compile(r"^(?:Then )?the stock is split equally into (\d+) \w+ and it keeps one share \(rounded down\)\.$", re.I),
     lambda x, m: x // int(m[0]) if int(m[0]) else None),
    (re.compile(r"^(?:Then )?the stock is doubled by merging with a sister \w+\.$", re.I), lambda x, m: x * 2),
    (re.compile(r"^(?:Then )?it ships (\d+) (\w+) per day for (\d+) days?\.$", re.I),
     lambda x, m: x - int(m[0]) * int(m[2])),
    (re.compile(r"^(?:Then )?it receives (\d+) deliver(?:y|ies) of (\d+) (\w+) each\.$", re.I),
     lambda x, m: x + int(m[0]) * int(m[1])),
]
_QUESTION = re.compile(r"^How many (\w+) does it have at the end\?$", re.I)


def solve_inventory(task: Task) -> FastResult | None:
    parts = core_sentences(task.instruction)
    if len(parts) < 2:
        return None
    start = _START.match(parts[0])
    question = _QUESTION.match(parts[-1])
    if not start or not question or question.group(1).lower() != start.group(3).lower():
        return None
    value = int(start.group(2))
    for sentence in parts[1:-1]:
        for pattern, apply in _OPS:
            m = pattern.match(sentence)
            if m:
                value = apply(value, m.groups())
                break
        else:
            return None
        if value is None:
            return None
    return FastResult("answer", str(value), "inventory template")


# ------------------------------------------------------------------- data: csv units

_CSV_DECL = re.compile(r"^The file ([\w./-]+\.csv) has columns ([\w,]+) \((\d+) rows\)\.$")
_CSV_UNITS = re.compile(
    r"^How many units of product '([^']+)' were sold in months (\d+) through (\d+) inclusive\?$"
)


def solve_csv_units(task: Task) -> FastResult | None:
    parts = core_sentences(task.instruction)
    if len(parts) != 2:
        return None
    decl, question = _CSV_DECL.match(parts[0]), _CSV_UNITS.match(parts[1])
    if not decl or not question:
        return None
    path = _resolve(task, decl.group(1))
    if path is None or not path.is_file():
        return FastResult("infeasible", reason=f"file {decl.group(1)} not found")
    reader = csv.DictReader(io.StringIO(read_text(path)))
    rows = list(reader)
    columns = {name.strip().casefold(): name for name in (reader.fieldnames or [])}
    needed = ("product", "month", "units")
    missing = [c for c in needed if c not in columns]
    if missing:
        return FastResult("infeasible", reason=f"columns missing: {missing}")
    col = {c: columns[c] for c in needed}
    product, lo, hi = question.group(1), int(question.group(2)), int(question.group(3))
    total = 0
    for row in rows:
        try:
            month, units = int(row[col["month"]]), int(row[col["units"]])
        except (TypeError, ValueError):
            return None  # dirty data: let the model reason about it
        if (row[col["product"]] or "").strip() == product and lo <= month <= hi:
            total += units
    return FastResult("answer", str(total), "csv units template")


# ------------------------------------------------------------------- search: markdown kb

_KB_DECL = re.compile(r"^The directory ([\w./-]+?)/? contains a small knowledge base of markdown files\.$")
_Q_BORN = re.compile(r"^In which year was (.+?) born\?$")
_Q_HQ = re.compile(r"^In which city is the headquarters of (.+?)'s employer\?$")


def _lookup(kb: dict, corpus: str, name: str, field: str) -> tuple[str | None, str]:
    entries = kb.get(norm(name))
    if not entries:
        if norm(name) not in corpus:
            return None, f"entity '{name}' is not in the knowledge base"
        return None, f"entity '{name}' has no profile"
    values = {e.get(field) for e in entries}
    if None in values:
        return None, f"'{name}' has no field '{field}'"
    if len(values) != 1:
        return None, f"ambiguous: {len(entries)} profiles named '{name}'"
    return values.pop(), ""


def solve_kb(task: Task) -> FastResult | None:
    parts = core_sentences(task.instruction)
    if len(parts) != 2:
        return None
    decl = _KB_DECL.match(parts[0])
    born, hq = _Q_BORN.match(parts[1]), _Q_HQ.match(parts[1])
    if not decl or not (born or hq):
        return None
    directory = _resolve(task, decl.group(1))
    if directory is None or not directory.is_dir():
        return FastResult("infeasible", reason="knowledge base directory not found")
    files = [p for p in list_files(directory) if p.suffix.lower() == ".md"]
    kb = parse_kb(files)
    if not kb or len(kb) < len(files) * 0.9:
        return None  # unusual format: let the model read it
    corpus = norm(" ".join(read_text(p, 200_000) for p in files))
    if born:
        value, why = _lookup(kb, corpus, born.group(1), "born")
        return FastResult("answer", value, "kb born") if value else FastResult("infeasible", reason=why)
    employer, why = _lookup(kb, corpus, hq.group(1), "employer")
    if not employer:
        return FastResult("infeasible", reason=why)
    city, why = _lookup(kb, corpus, employer, "headquarters")
    return FastResult("answer", city, "kb employer hq") if city else FastResult("infeasible", reason=why)


SOLVERS = (solve_inventory, solve_csv_units, solve_kb)


def solve(task: Task) -> FastResult | None:
    if task.is_coding:
        return None
    for solver in SOLVERS:
        try:
            result = solver(task)
        except (OSError, ValueError, KeyError, IndexError, ZeroDivisionError):
            result = None
        if result is not None:
            return result
    return None
