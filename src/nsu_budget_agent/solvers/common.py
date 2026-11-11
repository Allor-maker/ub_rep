"""Shared types and helpers for LLM-backed solvers."""

from __future__ import annotations

import re
from dataclasses import dataclass

UNSOLVABLE = "UNSOLVABLE"


@dataclass
class SolveResult:
    status: str  # "answer" | "infeasible" | "failed"
    answer: str = ""
    reason: str = ""
    files: dict | None = None  # coding: relative path -> new content (already verified by tests)
    code: str = ""  # the script that produced the answer (used by the verification judge)


def extract_code(text: str) -> str | None:
    blocks = re.findall(r"```(?:python3?|py)?[ \t]*\n(.*?)```", text, re.S)
    if blocks:
        return max(blocks, key=len)
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "def ", "print(")) and "\n" in stripped:
        return stripped  # an unfenced script
    return None


def says_unsolvable(text: str) -> bool:
    return text.strip().strip("`*. ").upper() == UNSOLVABLE


def normalize_answer(raw: str, check: str = "") -> str:
    """Last printed line -> bare value (no quotes, trailing period; '.0' dropped for exact_int only).

    Decimal formatting is part of some answers ("exactly one decimal place"), so a
    printed '12.0' is kept unless the task's check says the answer is an integer."""
    value = raw.strip().strip("`").strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        value = value[1:-1].strip()
    value = re.sub(r"^(?:final\s+)?answer\s*[:=]\s*", "", value, flags=re.I)
    if value.endswith("."):
        value = value[:-1].rstrip()
    if check == "exact_int" and re.fullmatch(r"-?\d+\.0+", value):
        value = value.split(".")[0]
    return value[:2000]


def spotlight(data: str) -> str:
    """Spotlighting (delimiting): wrap untrusted file content in tags the system prompt names
    as pure data; a closing tag planted inside the data is defused so it cannot end the block."""
    safe = data.replace("</untrusted_files", "</ untrusted_files").replace("<untrusted_files", "< untrusted_files")
    return f"<untrusted_files>\n{safe}\n</untrusted_files>"


_FORMATS = {
    "exact_int": re.compile(r"-?\d+"),
    "exact_decimal": re.compile(r"-?\d+(?:\.\d+)?"),
}


_ASKS_FRACTION = re.compile(r"\b(?:reduced\s+fraction|as\s+a\s+fraction|fraction\s+p/q|answer\s+p/q)\b", re.I)
_ASKS_DECIMALS = re.compile(r"\b(\d|one|two|three|four)\s+decimal(?:s|\s+places?)?\b", re.I)
_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4}
# A dict repr, or a list/tuple repr holding quoted strings: a printed row or key list, not a value.
# Plain number pairs like (4,9) or [1, 2] are legitimate answer forms and pass.
_DUMP = re.compile(r"\{\s*['\"].*:.*\}|[\[(]\s*['\"].*['\"]\s*[\])]")


def format_problem(answer: str, check: str, instruction: str = "") -> str | None:
    """Why the answer cannot be valid for the task's declared check or for the form the task text
    explicitly asks for (a reduced fraction, N decimals), or None if it can.

    The form check exists because a generic line like "Answer with one exact integer unless
    otherwise specified" made the model print only the numerator of a requested fraction."""
    pattern = _FORMATS.get(check)
    if pattern is not None and not pattern.fullmatch(answer):
        return f"the task expects a {check.removeprefix('exact_')} number, but the script printed {answer[:80]!r}"
    if len(answer) > 200:
        return "the last printed line is too long to be a final answer (debug output?)"
    if answer in ("None", "null", "nan", "NaN"):
        # A lookup that found nothing (wrong key, filter or layout), not an answer.
        return (f"the script printed {answer}: the value was not found (check keys, filters and the real file "
                "layout in the stdout below); if it truly cannot be determined, print UNSOLVABLE")
    if _DUMP.fullmatch(answer):
        # Exploration output (rows printed to look at the data) taken as the answer.
        return (f"the last line is a printed Python dict/list {answer[:60]!r}, not the requested value "
                "(debug output?); use the data shown below and print only the final answer")
    if check != "exact_int" and instruction:
        if _ASKS_FRACTION.search(instruction) and not re.fullmatch(r"-?\d+/\d+", answer):
            return (f"the task asks for a reduced fraction p/q (e.g. 3/1000), but the script printed {answer[:80]!r} "
                    "(if the exact value is a whole number, printing the integer is right)")
        decimals = _ASKS_DECIMALS.search(instruction)
        if decimals:
            n = int(_WORDS.get(decimals.group(1).lower(), decimals.group(1)))
            if not re.fullmatch(rf"-?\d+\.\d{{{n}}}", answer):
                return f"the task asks for exactly {n} decimal places, but the script printed {answer[:80]!r}"
    return None


def last_line(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""
