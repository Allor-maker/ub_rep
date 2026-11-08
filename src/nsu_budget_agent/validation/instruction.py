"""Independent binding of the closed local grammar to a proposed operation."""

import re

from nsu_budget_agent.contracts.models import TaskSpec, ToolCall
from nsu_budget_agent.solvers.plans import _question
from nsu_budget_agent.solvers.sources import selected_sources, simple_entity


def instruction_matches(task: TaskSpec, call: ToolCall) -> bool:
    # Only the non-semantic, explicitly allowed execution/output suffix policy
    # is shared. Operation/condition extraction below is separate from plan().
    try:
        text = _question(task.instruction)
    except ValueError:
        return False
    args = call.arguments
    if call.tool_name == "local_math":
        expression = re.fullmatch(
            r"(?:Compute|Calculate|Evaluate)\s+([\d+*/%(). \-]+)(?:\s+Return only the result\.)?",
            text,
            re.I,
        )
        if expression:
            return args == {"expression": expression[1].rstrip(".").strip()}
        match = re.fullmatch(
            r"A warehouse starts with (\d+) boxes(.+?)(?:How many boxes does it have at the end|How many boxes are left|What is the final stock|What is its final stock)\?",
            text,
            re.I,
        )
        if not match:
            return False
        body = match[2]
        operations = []
        stock_patterns = (
            (r"(?:It )?ships (\d+) boxes per day for (\d+) days", "ship"),
            (r"(?:It )?receives (\d+) deliveries of (\d+) boxes each", "delivery"),
            (
                r"(?:(?:the stock is|It is) )?split equally into (\d+) warehouses(?: and (?:it )?keeps|, keeping) one share \(rounded down\)",
                "floor_div",
            ),
            (
                r"(?:(?:the stock is|It is) )?doubled(?: by merging with a sister warehouse)?",
                "double",
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
            body = re.sub(r"^then\s+", "", body.lstrip(" .,\n\r"), flags=re.I)
            for pattern, operation in stock_patterns:
                part = re.match(pattern, body, re.I)
                if part:
                    values = [int(v) for v in part.groups()]
                    if operation in ("ship", "delivery"):
                        op, value = ("sub" if operation == "ship" else "add"), values[0] * values[1]
                    elif operation == "double":
                        op, value = "mul", 2
                    else:
                        op, value = operation, values[0]
                    operations.append({"op": op, "value": value})
                    body = body[part.end() :]
                    break
            else:
                return False
            if len(operations) > 64:
                return False
        return args == {"initial": int(match[1]), "operations": operations}
    if call.tool_name == "lookup_markdown":
        text = re.sub(
            r"^The directory \S+(?: in workspace/)? contains a small knowledge base of markdown files\.\s*",
            "",
            text,
            flags=re.I,
        )
        search_patterns: tuple[tuple[str, list[str], str], ...] = (
            (
                r"In which city is the headquarters of (.+?)'s employer\?",
                ["Employer"],
                "Headquarters",
            ),
            (r"In which year was (.+?)'s employer founded\?", ["Employer"], "Founded"),
            (r"In which year was (.+?)'s mentor born\?", ["Mentor"], "Born"),
            (r"In which year was (.+?) born\?", [], "Born"),
        )
        for pattern, chain, field in search_patterns:
            found = re.fullmatch(pattern, text, re.I)
            if found and simple_entity(found[1]):
                return args == {
                    "paths": [
                        a.path.as_posix()
                        for a in selected_sources(task)
                        if a.path.suffix.casefold() == ".md"
                    ],
                    "entity": found[1],
                    "chain": chain,
                    "field": field,
                }
        return False
    if call.tool_name != "query_csv":
        return False
    header = re.fullmatch(
        r"The file (\S+\.csv)(?: in workspace/)? has columns ([\w,]+) \((\d+) rows\)\. (.+)",
        text,
        re.I,
    )
    if header:
        declared = {"expected_rows": int(header[3]), "expected_columns": header[2].split(",")}
        text = header[4]
        order = re.fullmatch(
            r"How many orders of product '([^']+)' came from region '([^']+)'\?", text, re.I
        )
        if order:
            expected = {
                "path": header[1],
                "aggregate": "count_distinct",
                "columns": ["order_id"],
                "filters": [
                    {"column": "product", "op": "eq", "value": order[1]},
                    {"column": "region", "op": "eq", "value": order[2]},
                ],
                **declared,
            }
            return args == expected
        months = re.fullmatch(
            r"(How many units of product '([^']+)' were sold|What is the total revenue \(units multiplied by price\) for product '([^']+)')(?: in region '([^']+)' during| in) months (\d+) through (\d+) inclusive\?",
            text,
            re.I,
        )
        if not months:
            return False
        filters = [
            {"column": "product", "op": "eq", "value": months[2] or months[3]},
            {"column": "month", "op": "ge", "value": months[5]},
            {"column": "month", "op": "le", "value": months[6]},
        ]
        if months[4] is not None:
            filters.append({"column": "region", "op": "eq", "value": months[4]})
        return (
            args
            == {
                "path": header[1],
                "aggregate": "sum" if months[2] else "product_sum",
                "columns": ["units"] if months[2] else ["units", "price"],
                "filters": filters,
                **declared,
            }
            and 1 <= int(months[5]) <= int(months[6]) <= 12
        )
    simple = re.fullmatch(
        r"In (\S+\.csv), (?:what is the sum of (\w+) where (\w+) equals '([^']+)'|how many rows where (\w+) equals '([^']+)')\?",
        text,
        re.I,
    )
    if not simple:
        return False
    return args == {
        "path": simple[1],
        "aggregate": "sum" if simple[2] else "count_rows",
        "columns": [simple[2]] if simple[2] else [],
        "filters": [
            {"column": simple[3] or simple[5], "op": "eq", "value": simple[4] or simple[6]}
        ],
    }
