"""Free infeasibility signals for tasks the fast path did not recognize.

A question about a person/organisation that is mentioned NOWHERE in the task's knowledge
base cannot be answered from the data. Names are capitalized multi-word phrases of the
instruction (boilerplate removed); a phrase counts as present if it occurs anywhere in
the corpus (case/whitespace-insensitive). The runner applies this signal only to tasks
whose reward is not high: for valuable tasks a cheap LLM check is worth its tokens.
"""

from __future__ import annotations

import re

from ..context import list_files, norm, read_text
from ..tasks import Task
from .fastpath import core_sentences

_NAME = re.compile(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+")
_LEADING = {
    "in", "which", "what", "who", "whom", "how", "the", "use", "find", "work", "when", "write", "compute",
    "run", "fix", "consult", "locate", "from", "read", "return", "follow", "answer", "only", "a", "an",
    "its", "each", "then", "is", "was", "does", "did", "for", "of", "to", "and", "or", "if", "by",
}


def candidate_names(instruction: str) -> list[str]:
    names: list[str] = []
    for match in _NAME.finditer(" ".join(core_sentences(instruction))):
        words = match.group().split()
        while words and words[0].lower() in _LEADING:
            words = words[1:]
        if len(words) >= 2:
            names.append(" ".join(words))
    return names


def missing_names(task: Task) -> list[str]:
    md = [p for p in list_files(task.workspace) if p.suffix.lower() in (".md", ".markdown")]
    if len(md) < 3:
        return []  # not a knowledge-base task
    corpus = norm(" ".join(read_text(p, 200_000) for p in md))
    return [name for name in candidate_names(task.instruction) if norm(name) not in corpus]
