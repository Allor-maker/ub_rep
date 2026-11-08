"""Bounded semantic families for independently verified LLM proposals.

Only fully consumed instructions are recognized. All values, names, periods
and paths come from public input; there are no task IDs or answer tables.
"""

import math
import re
from pathlib import PurePosixPath
from typing import cast

from nsu_budget_agent.contracts.models import JsonObject, TaskSpec, ToolCall, parse_rel_path
from nsu_budget_agent.solvers.sources import selected_sources, simple_entity


def without_output_policy(text: str) -> str:
    from nsu_budget_agent.solvers.language import core_text

    text = core_text(text)
    policies = (
        r"Work only inside the current task workspace\.",
        r"Write ONLY the final answer to answer\.txt next to task\.json, with no explanation\.",
        r"(?:Give|Return) (?:the )?(?:final |exact )?integer(?: total)? only\.",
        r"(?:Give|Return) (?:the )?exact integer total\.",
        r"(?:Return one integer|Output just (?:that integer|the count)|Answer (?:with|as) an integer)\.",
        r"(?:Reply with the four-digit year|Output only the year|Output the city name only)\.",
    )
    text = text.strip()
    while True:
        for policy in policies:
            found = re.search(r"(?:^|\s)" + policy + r"$", text, re.I)
            if found:
                text = text[: found.start()].strip()
                break
        else:
            return text


def integer(text: str) -> int:
    words = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split()
    return words.index(text.lower()) if text.lower() in words else int(text)


def discrete_question(instruction: str) -> tuple[str, tuple[int, ...]] | None:
    from nsu_budget_agent.solvers.language import discrete_spec

    extended = discrete_spec(instruction)
    if extended is not None:
        return extended
    text = without_output_policy(instruction)
    try:
        m = re.fullmatch(
            r"(?:Find|Calculate|What is) the sum of the first (\w+) prime numbers(?:, starting with 2)?[.?]",
            text,
            re.I,
        )
        if m and 1 <= integer(m[1]) <= 1000:
            return "prime_sum", (integer(m[1]),)
        m = re.fullmatch(
            r"What is the smallest positive integer divisible by ([0-9, and]+) with no remainder\?",
            text,
            re.I,
        )
        if m:
            pieces = re.split(r",\s*(?:and\s+)?|\s+and\s+", m[1])
            if 2 <= len(pieces) <= 16 and all(re.fullmatch(r"\d{1,6}", p.strip()) for p in pieces):
                values = tuple(int(p) for p in pieces)
                if all(v > 0 for v in values):
                    return "lcm", values
        m = re.fullmatch(
            r"A (?:small )?(?:theatre|theater|auditorium) has (\d+) rows\. The first row has (\d+) seats, and each subsequent row has (\d+) more seats than the preceding row\. How many seats are there in total across all (\d+) rows\?",
            text,
            re.I,
        )
        if m and m[1] == m[4] and all(len(p) <= 6 for p in m.groups()) and int(m[1]) > 0:
            return "series", tuple(map(int, m.groups()[:3]))
        m = re.fullmatch(
            r"A (?:lock )?code has exactly (\d+) different digits chosen from ([0-9, and]+)\. The order matters and a code may begin with zero\. How many distinct codes are possible\?",
            text,
            re.I,
        )
        if m:
            pieces = re.split(r",\s*(?:and\s+)?|\s+and\s+", m[2])
            if all(re.fullmatch(r"[0-9]", p.strip()) for p in pieces):
                digits = tuple(int(p) for p in pieces)
                if len(digits) == len(set(digits)) and 1 <= int(m[1]) <= len(digits) <= 10:
                    return "arrangements", (len(digits), int(m[1]))
    except (ValueError, OverflowError):
        return None
    return None


def discrete_answer(kind: str, values: tuple[int, ...]) -> str:
    if kind == "combinations":
        return str(math.comb(*values))
    if kind == "multiples":
        return str(sum(n for n in range(1, values[0] + 1) if n % values[1] == 0))
    if kind == "prime_sum":
        primes: list[int] = []
        candidate = 2
        while len(primes) < values[0]:
            if all(candidate % p for p in primes if p * p <= candidate):
                primes.append(candidate)
            candidate += 1
        return str(sum(primes))
    if kind == "lcm":
        return str(math.lcm(*values))
    if kind == "series":
        count, initial, delta = values
        return str(count * (2 * initial + (count - 1) * delta) // 2)
    if kind == "arrangements":
        return str(math.perm(*values))
    raise ValueError("unsupported discrete semantics")


def routed_data(task: TaskSpec) -> tuple[ToolCall, tuple[PurePosixPath, ...]] | None:
    """Closed relational/CSV clause grammars; never a keyword-only certificate."""
    text = without_output_policy(task.instruction)
    sources = selected_sources(task)
    allowed = {a.path for a in sources}
    file = r"(?P<path>(?:workspace/)?[A-Za-z0-9_/-]+\.csv)"
    value = r"[A-Za-z][A-Za-z0-9_-]{0,39}"
    prefix = re.fullmatch(r"(?:From|Using|In) " + file + r", (.+)", text, re.I)
    body = prefix[2] if prefix else None
    path_text = prefix["path"] if prefix else None
    if body is None:
        prefix = re.fullmatch(
            r"The " + file + r" file lists units, price and status\. (.+)", text, re.I
        )
        if prefix:
            path_text, body = prefix["path"], prefix[2]
    if body is not None and path_text is not None:
        path = parse_rel_path(path_text.removeprefix("workspace/"))
        if path not in allowed:
            return None
        # Every trailing sentence must be an explicit, compatible clarification.
        chunks = re.split(r"(?<=[.?])\s+", body)
        main, rest = chunks[0], chunks[1:]
        aggregate = None
        filters: list[dict[str, str]] = []
        selected_status = None
        m = re.fullmatch(
            r"(?:calculate|compute|find) (?:the )?(?:exact )?(?:total )?revenue \(units (?:multiplied by|times) price\) from (?:records|rows|orders) where (.+)\.",
            main,
            re.I,
        )
        if m:
            aggregate = "product_sum"
            clauses = re.split(r",\s*(?:and\s+)?|\s+and\s+", m[1])
            for clause in clauses:
                part = re.fullmatch(
                    r"(product|region|status) (?:is|equals) ('?)(" + value + r")\2", clause, re.I
                )
                if not part or any(f["column"] == part[1].lower() for f in filters):
                    return None
                filters.append({"column": part[1].lower(), "op": "eq", "value": part[3]})
        m = re.fullmatch(
            r"how many (?:order rows|rows) for product ("
            + value
            + r") in the ("
            + value
            + r") region have status ("
            + value
            + r")\?",
            main,
            re.I,
        )
        if m:
            aggregate = "count_rows"
            filters = [
                {"column": k, "op": "eq", "value": v}
                for k, v in zip(("product", "region", "status"), m.groups())
            ]
        m = re.fullmatch(
            r"add up units for the ("
            + value
            + r")-region ("
            + value
            + r") orders that are marked ("
            + value
            + r") and whose month is ([0-9, or]+)\.",
            main,
            re.I,
        )
        if m:
            pieces = re.split(r",\s*(?:or\s+)?|\s+or\s+", m[4])
            if not all(re.fullmatch(r"\d{1,2}", p.strip()) for p in pieces):
                return None
            months = sorted(int(p) for p in pieces)
            if (
                not months
                or months != list(range(months[0], months[-1] + 1))
                or not 1 <= months[0] <= months[-1] <= 12
            ):
                return None
            aggregate = "sum"
            filters = [
                {"column": k, "op": "eq", "value": v}
                for k, v in zip(("region", "product", "status"), m.groups()[:3])
            ]
            filters += [
                {"column": "month", "op": "ge", "value": str(months[0])},
                {"column": "month", "op": "le", "value": str(months[-1])},
            ]
        m = re.fullmatch(
            r"What is the revenue from ("
            + value
            + r")-region ("
            + value
            + r") orders in months (\d{1,2}) through (\d{1,2}) inclusive that have status ("
            + value
            + r")\?",
            main,
            re.I,
        )
        if m and 1 <= int(m[3]) <= int(m[4]) <= 12:
            aggregate = "product_sum"
            filters = [
                {"column": k, "op": "eq", "value": v}
                for k, v in zip(("region", "product", "status"), (m[1], m[2], m[5]))
            ]
            filters += [
                {"column": "month", "op": "ge", "value": m[3]},
                {"column": "month", "op": "le", "value": m[4]},
            ]
        if aggregate is None:
            return None
        selected_status = next((f["value"] for f in filters if f["column"] == "status"), None)
        for sentence in rest:
            if re.fullmatch(r"Do not include any other status or month\.", sentence, re.I):
                if selected_status is None or not any(f["column"] == "month" for f in filters):
                    return None
                continue
            if re.fullmatch(
                r"Compute revenue as units times price and return its exact integer sum\.",
                sentence,
                re.I,
            ):
                if aggregate != "product_sum":
                    return None
                continue
            if (
                re.match(r"Count orders rather than summing units", sentence, re.I)
                and aggregate != "count_rows"
            ):
                return None
            sentence = re.sub(
                r"^Count orders rather than summing units, and ", "", sentence, flags=re.I
            )
            excluded = re.fullmatch(
                r"(?:Exclude|do not count) ([A-Za-z, orand]+) (?:records even if they have matching product and region|rows)\.",
                sentence,
                re.I,
            )
            if excluded is None:
                excluded = re.fullmatch(r"([A-Za-z]+) orders must not count\.", sentence, re.I)
            if excluded is None or selected_status is None:
                return None
            statuses = re.split(r",\s*|\s+(?:or|and)\s+", excluded[1].lower())
            statuses = [v.removeprefix("or ").removeprefix("and ").strip() for v in statuses]
            if not statuses or any(
                not v.isalpha() or v == selected_status.lower() for v in statuses
            ):
                return None
        filters.sort(key=lambda f: (f["column"], f["op"], f["value"]))
        return ToolCall(
            tool_name="query_csv",
            arguments=cast(
                JsonObject,
                {
                    "path": path.as_posix(),
                    "aggregate": aggregate,
                    "columns": []
                    if aggregate == "count_rows"
                    else (["units", "price"] if aggregate == "product_sum" else ["units"]),
                    "filters": filters,
                },
            ),
        ), (path,)

    # Explicit source scope must agree in the opening and closing clauses.
    scope = re.search(r"workspace/([A-Za-z0-9_/-]+)/", text)
    if scope is None:
        return None
    directory = scope[1]
    tail = rf" Use only the Markdown documents provided under workspace/{re.escape(directory)}/\. Do not use the internet\."
    if not re.search(tail + "$", text, re.I):
        return None
    text = without_output_policy(re.sub(tail + "$", "", text, flags=re.I))
    directory_pattern = re.escape("workspace/" + directory + "/")
    patterns = (
        (
            r"Using (?:the files in )?"
            + directory_pattern
            + r", identify (?P<entity>.+?)'s employer and then its parent organization\. What year was that parent organization founded\?",
            ["Employer", "Parent organization"],
            "Founded",
        ),
        (
            r"According to "
            + directory_pattern
            + r", (?P<entity>.+?) works for a firm that names the coordinator of its annual research summit\. In which city did that coordinator host its (?P<year>\d{4}) summit\?",
            ["Employer", "Research summit coordinator"],
            "{year} summit host city",
        ),
        (
            r"Using only "
            + directory_pattern
            + r", first find the mentor of (?P<entity>.+?), then find the company employing that mentor\. In which year was that company founded\?",
            ["Mentor", "Employer"],
            "Founded",
        ),
        (
            r"The lab associated with (?P<entity>.+?) has a named program coordinator\. Which MONTH did that coordinator launch the (?P<initiative>[A-Za-z][A-Za-z ]{0,60}) initiative\? Find the answer using "
            + directory_pattern
            + r" and output only the English month name\.",
            ["Lab", "Program coordinator"],
            "{initiative} initiative launch month",
        ),
    )
    for pattern, chain, field in patterns:
        m = re.fullmatch(pattern, text, re.I)
        if m and simple_entity(m["entity"]):
            paths = tuple(
                sorted(
                    p
                    for p in allowed
                    if p.parent.as_posix() == directory and p.suffix.casefold() == ".md"
                )
            )
            if not 0 < len(paths) <= 32:
                return None
            field = field.format(**m.groupdict())
            return ToolCall(
                tool_name="lookup_markdown",
                arguments=cast(
                    JsonObject,
                    {
                        "paths": [p.as_posix() for p in paths],
                        "entity": m["entity"],
                        "chain": chain,
                        "field": field,
                    },
                ),
            ), paths
    return None
