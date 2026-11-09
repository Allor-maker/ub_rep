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


def last_line(stdout: str) -> str:
    lines = [line for line in stdout.splitlines() if line.strip()]
    return lines[-1] if lines else ""
