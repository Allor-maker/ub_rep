"""Local bounded profile retrieval; original documents remain the evidence."""

import re
from collections.abc import Callable

from nsu_budget_agent.contracts.models import ErrorInfo, OperationFailed
from nsu_budget_agent.solvers.sources import sensitive_text


def retrieve_profiles(
    documents: dict[str, str], query: str, check: Callable[[], None]
) -> dict[str, str]:
    from nsu_budget_agent.orchestration.decision import injection_marker

    if not 0 < len(documents) <= 128 or len(query) > 10000:
        raise ValueError("retrieval bounds")
    profiles = {}
    identities = set()
    for path, text in documents.items():
        check()
        if injection_marker(text) or sensitive_text(text):
            raise OperationFailed(ErrorInfo(code="denied", message="unsafe source corpus"))
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or not re.fullmatch(r"# [^#]+", lines[0]):
            raise ValueError("unstructured profile")
        name = lines[0][2:].strip()
        if name.casefold() in identities:
            raise ValueError("ambiguous corpus identity")
        identities.add(name.casefold())
        values = []
        fields = set()
        for line in lines[1:]:
            match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9 ]{0,127}):\s*(\S.*)", line)
            if not match or match[1].casefold() in fields:
                raise ValueError("unstructured corpus property")
            fields.add(match[1].casefold())
            values.append(match[2].strip())
        profiles[name] = (path, values)
    selected = {
        name
        for name in profiles
        if re.search(r"(?<![\w])" + re.escape(name) + r"(?![\w])", query, re.I)
    }
    if not selected:
        raise OperationFailed(
            ErrorInfo(code="tool.retrieval_no_seed", message="no explicit profile name")
        )
    frontier = set(selected)
    for _ in range(4):
        check()
        frontier = {
            value for name in frontier for value in profiles[name][1] if value in profiles
        } - selected
        selected.update(frontier)
        if len(selected) > 16:
            raise OperationFailed(
                ErrorInfo(code="resource_limit", message="retrieval fanout bound")
            )
    result = {profiles[name][0]: documents[profiles[name][0]] for name in sorted(selected)}
    return result


def profile_context(documents: dict[str, str], query: str) -> dict[str, str]:
    """Property excerpts for the model; full original text was hash-checked first.

    The Validator always reopens original sources, never these model excerpts.
    Whole-corpus injection checks happen before irrelevant properties are omitted.
    """
    ordinary = {
        "employer",
        "mentor",
        "parent organization",
        "lab",
        "program coordinator",
        "research summit coordinator",
        "born",
        "founded",
        "headquarters",
        "email",
        "department",
        "first foreign subsidiary acquisition year",
    }
    result = {}
    for path, text in documents.items():
        lines = text.splitlines()
        chosen = []
        for line in lines:
            if line.startswith("# "):
                chosen.append(line)
                continue
            field, separator, _ = line.partition(":")
            key = field.strip().casefold()
            year = re.fullmatch(r"(\d{4}) summit host city", key)
            initiative = re.fullmatch(r"(.+) initiative launch month", key)
            if separator and (
                key in ordinary
                or (year and year[1] in query)
                or (initiative and initiative[1] in query.casefold())
            ):
                chosen.append(line)
        result[path] = "\n".join(chosen)
    if sum(len(text.encode("utf-8")) for text in result.values()) > 16384:
        raise ValueError("bounded source policy exceeded")
    return result
