"""Explicit task-source selection; credentials are never model/tool inputs."""

import re
from pathlib import PurePosixPath

from nsu_budget_agent.contracts.models import ArtifactRef, TaskSpec


def simple_entity(entity: str) -> bool:
    words = re.findall(r"[A-Za-z]+", entity)
    qualifiers = set(
        (
            "and or nor not but except unless without excluding only either neither with who whose "
            "which where if then rather than versus vs including of from in at for by to the "
            "according is are was were born based located living working before after since "
            "until while no never else instead"
        ).split()
    )
    return (
        bool(words)
        and re.fullmatch(r"[A-Za-z][A-Za-z .'-]{0,79}", entity) is not None
        and all(word[0].isupper() and word.casefold() not in qualifiers for word in words)
        and "'s" not in entity
    )


def sensitive(path: PurePosixPath) -> bool:
    return any(
        re.search(
            r"(?:^|[^a-z])(?:credentials?|secrets?|passwords?|tokens?|api[_-]?keys?|private[_-]?keys?)(?:$|[^a-z])",
            part.casefold(),
        )
        is not None
        or part.startswith(".")
        or part.casefold().endswith((".pem", ".key", ".p12"))
        for part in path.parts
    )


def sensitive_text(text: str) -> bool:
    return (
        re.search(
            r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|"
            r"\b(?:api[_-]?key|access[_-]?token|client[_-]?secret|password|secret[_-]?key|token)\s*(?:[=:,]|\r?$)",
            text,
            re.I | re.M,
        )
        is not None
    )


def selected_sources(task: TaskSpec) -> tuple[ArtifactRef, ...]:
    """Only explicit paths/directories or the closed local Markdown question.

    The original artifact manifest remains intact for A5 integrity checks.
    """
    result = []
    for artifact in task.artifacts:
        raw = artifact.path.as_posix()
        explicit = (
            re.search(r"(?<![\w/.-])" + re.escape(raw) + r"(?![\w/-]|\.[\w])", task.instruction)
            is not None
        )
        scoped = (
            any(
                directory + "/*.md" in task.instruction
                for directory in (artifact.path.parent.as_posix(),)
            )
            and artifact.path.suffix == ".md"
        )
        directory = re.search(r"The directory (\S+?)(?: in workspace/)? contains", task.instruction)
        scoped |= bool(directory and artifact.path.is_relative_to(directory[1].rstrip("/")))
        implicit_markdown = (
            re.fullmatch(
                r"In which (?:year was|city is the headquarters of) (.+?)(?: born|'s (?:employer(?: founded)?|mentor born))\?",
                task.instruction.strip(),
                re.I,
            )
            is not None
            and artifact.path.suffix == ".md"
        )
        if explicit and sensitive(artifact.path):
            raise ValueError("sensitive source is forbidden")
        if (explicit or scoped or implicit_markdown) and not sensitive(artifact.path):
            if artifact.path.suffix.casefold() not in (".csv", ".md", ".txt"):
                raise ValueError("source format is unavailable")
            result.append(artifact)
    return tuple(result)
