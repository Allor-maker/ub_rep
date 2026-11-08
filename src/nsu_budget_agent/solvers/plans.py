"""Small, explicit instruction grammar. Unconsumed task text is unsupported."""

import re
from dataclasses import dataclass
from pathlib import PurePosixPath

from nsu_budget_agent.contracts.models import JsonValue, TaskSpec, ToolCall, parse_rel_path

from .sources import sensitive, simple_entity


class Unsupported(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    kind: str
    call: ToolCall


_SUFFIXES = (
    r"Compute (?:it )?with (?:the bash tool \(python3 -c\)|python3)\.",
    r"Use bash with python3 to compute it\.",
    r"Use python3 to compute an exact result from the available data\.",
    r"Use the file tools or bash \(ls, grep, cat\) to find it\.",
    r"Use the available files to find it; do not infer unsupported facts\.",
    r"Work only inside the current (?:task workspace|directory)\.",
    r"Write ONLY the final answer to answer\.txt next to task\.json, with no explanation\.",
    r"When you are done, write ONLY the final answer to the file answer\.txt in the current directory "
    r"\(for example with bash: echo -?\d{1,30} > answer\.txt\)\.",
)


def _question(instruction: str) -> str:
    if len(instruction) > 10_000:
        raise Unsupported("instruction exceeds grammar limit")
    question, separator, rest = instruction.partition("?")
    if not separator:
        # Arithmetic has its own complete, anchored grammar.
        return instruction.strip()
    while rest.strip():
        rest = rest.strip()
        for pattern in _SUFFIXES:
            match = re.match(pattern, rest, re.IGNORECASE)
            if match:
                rest = rest[match.end() :]
                break
        else:
            raise Unsupported("unrecognized instruction suffix")
    return question.strip() + "?"


def _warehouse(text: str) -> ToolCall | None:
    match = re.fullmatch(
        r"A warehouse starts with (\d{1,30}) boxes(?P<body>.+?)"
        r"(?:How many boxes does it have at the end|How many boxes are left|"
        r"What is the final stock|What is its final stock)\?",
        text,
        re.I,
    )
    if not match:
        return None
    body = match["body"]
    operations: list[JsonValue] = []
    patterns = (
        (r"(?:It |it )?ships (\d+) boxes per day for (\d+) days", "ship"),
        (r"(?:It |it )?receives (\d+) deliveries of (\d+) boxes each", "delivery"),
        (
            r"(?:Then the stock is |The stock is |It is |it is )?split equally into (\d+) warehouses"
            r"(?: and (?:it )?keeps|, keeping) one share \(rounded down\)",
            "floor_div",
        ),
        (
            r"(?:Then the stock is |The stock is |It is |it is )?doubled(?: by merging with a sister warehouse)?",
            "mul2",
        ),
        (r"split into (\d+) equal shares, keeping one share \(rounded down\)", "floor_div"),
        (r"receives (\d+) new boxes", "add"),
        (r"discards (\d+) damaged boxes", "sub"),
        (r"sends (\d+) batches of (\d+) boxes", "ship"),
        (
            r"The remainder is divided equally among (\d+) sites, and this warehouse keeps one share \(rounded down\)",
            "floor_div",
        ),
        (r"receives an extra (\d+) boxes", "add"),
    )
    while body.strip(" .,\n\r"):
        body = body.lstrip(" .,\n\r")
        body = re.sub(r"^then\s+", "", body, flags=re.I)
        for pattern, operation in patterns:
            part = re.match(pattern, body, re.I)
            if part:
                values = [int(value) for value in part.groups()]
                if operation in ("ship", "delivery"):
                    op, value = ("sub" if operation == "ship" else "add"), values[0] * values[1]
                elif operation == "mul2":
                    op, value = "mul", 2
                else:
                    op, value = operation, values[0]
                operations.append({"op": op, "value": value})
                body = body[part.end() :]
                break
        else:
            raise Unsupported("unrecognized or incomplete warehouse operation")
        if len(operations) > 64:
            raise Unsupported("operation limit")
    return ToolCall(
        tool_name="local_math", arguments={"initial": int(match[1]), "operations": operations}
    )


def plan(task: TaskSpec) -> Plan:
    if task.type_hint == "coding" or task.check_hint == "unittest":
        raise Unsupported("coding and test execution are unavailable")
    text = _question(task.instruction)
    arithmetic = re.fullmatch(
        r"(?:Compute|Calculate|Evaluate)\s+([0-9+*/%(). \-]+)\.?(?:\s+Return only the result\.)?",
        text,
        re.I,
    )
    if arithmetic:
        return Plan(
            "math",
            ToolCall(
                tool_name="local_math", arguments={"expression": arithmetic[1].rstrip(".").strip()}
            ),
        )
    warehouse = _warehouse(text)
    if warehouse:
        return Plan("math", warehouse)
    search = (
        (
            r"In which city is the headquarters of (?P<entity>.+?)'s employer\?",
            ("Employer",),
            "Headquarters",
        ),
        (r"In which year was (?P<entity>.+?)'s employer founded\?", ("Employer",), "Founded"),
        (r"In which year was (?P<entity>.+?)'s mentor born\?", ("Mentor",), "Born"),
        (r"In which year was (?P<entity>.+?) born\?", (), "Born"),
    )
    kb_prefix = r"(?:The directory (?P<directory>\S+)(?: in workspace/)? contains a small knowledge base of markdown files\.\s*)?"
    for pattern, chain, field in search:
        match = re.fullmatch(kb_prefix + pattern, text, re.I)
        if match:
            if not simple_entity(match["entity"]):
                raise Unsupported("entity contains an unsupported qualification or alternative")
            directory = (
                parse_rel_path(match["directory"].rstrip("/")) if match["directory"] else None
            )
            paths: list[JsonValue] = [
                a.path.as_posix()
                for a in task.artifacts
                if a.path.suffix.casefold() == ".md"
                and not sensitive(a.path)
                and (directory is None or a.path.is_relative_to(directory))
            ]
            if not paths:
                raise Unsupported("no Markdown evidence")
            return Plan(
                "search",
                ToolCall(
                    tool_name="lookup_markdown",
                    arguments={
                        "paths": paths,
                        "entity": match["entity"],
                        "chain": [str(relation) for relation in chain],
                        "field": field,
                    },
                ),
            )
    file_prefix = re.fullmatch(
        r"The file (\S+\.csv)(?: in workspace/)? has columns ([\w,]+) \((\d+) rows\)\. (.+)",
        text,
        re.I,
    )
    expected_rows = None
    if file_prefix:
        path = parse_rel_path(file_prefix[1])
        question = file_prefix[4]
        expected_rows = int(file_prefix[3])
        columns = file_prefix[2].split(",")
    else:
        rows = re.fullmatch(
            r"In (\S+\.csv), how many rows where (\w+) equals '([^']+)'\?", text, re.I
        )
        if rows:
            return _csv_plan(
                task,
                parse_rel_path(rows[1]),
                {
                    "aggregate": "count_rows",
                    "columns": [],
                    "filters": [{"column": rows[2], "op": "eq", "value": rows[3]}],
                },
            )
        simple = re.fullmatch(
            r"In (\S+\.csv), what is the sum of (\w+) where (\w+) equals '([^']+)'\?", text, re.I
        )
        if not simple:
            raise Unsupported("no supported deterministic interpretation")
        path = parse_rel_path(simple[1])
        return _csv_plan(
            task,
            path,
            {
                "aggregate": "sum",
                "columns": [simple[2]],
                "filters": [{"column": simple[3], "op": "eq", "value": simple[4]}],
            },
        )
    filters = []
    match = re.fullmatch(
        r"How many orders of product '([^']+)' came from region '([^']+)'\?", question, re.I
    )
    if match:
        if columns.count("order_id") != 1 or len(columns) != len(set(columns)):
            raise Unsupported("orders require an explicitly declared order_id")
        aggregate = "count_distinct"
        values: list[str] = ["order_id"]
        filters = [
            {"column": "product", "op": "eq", "value": match[1]},
            {"column": "region", "op": "eq", "value": match[2]},
        ]
    else:
        match = re.fullmatch(
            r"(?:(How many units) of product '([^']+)' were sold|"
            r"(What is the total revenue \(units multiplied by price\)) for product '([^']+)')"
            r"(?: in region '([^']+)' during| in) months (\d+) through (\d+) inclusive\?",
            question,
            re.I,
        )
        if not match:
            raise Unsupported("unknown CSV question")
        aggregate, values = ("sum", ["units"]) if match[1] else ("product_sum", ["units", "price"])
        filters = [
            {"column": "product", "op": "eq", "value": match[2] or match[4]},
            {"column": "month", "op": "ge", "value": match[6]},
            {"column": "month", "op": "le", "value": match[7]},
        ]
        if not 1 <= int(match[6]) <= int(match[7]) <= 12:
            raise Unsupported("invalid month interval")
        if match[5] is not None:
            filters.append({"column": "region", "op": "eq", "value": match[5]})
    return _csv_plan(
        task,
        path,
        {
            "aggregate": aggregate,
            "columns": values,
            "filters": filters,
            "expected_rows": expected_rows,
            "expected_columns": columns,
        },
    )


def _csv_plan(task: TaskSpec, path: PurePosixPath, arguments: dict) -> Plan:
    if sensitive(path):
        raise Unsupported("sensitive CSV source is forbidden")
    if path not in {a.path for a in task.artifacts}:
        raise Unsupported("specified CSV is not an allowed artifact")
    return Plan(
        "data_analysis",
        ToolCall(tool_name="query_csv", arguments={"path": path.as_posix(), **arguments}),
    )
