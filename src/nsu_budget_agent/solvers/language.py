"""Small compositional domain grammars, never a keyword-only certification.

All material tokens must be consumed. Names, paths, operands and predicates come
from public input. Unsupported qualifications stay unsupported, including OR,
negation and additional numerical conditions unless explicitly represented.
"""

import json
import re
from collections.abc import Callable
from functools import lru_cache, wraps
from typing import TypeVar, cast

from nsu_budget_agent.contracts.models import JsonObject, JsonValue, TaskSpec, ToolCall

from .sources import selected_sources, simple_entity

_Key = TypeVar("_Key", str, TaskSpec)


def cached_call(function: Callable[[_Key], ToolCall | None]) -> Callable[[_Key], ToolCall | None]:
    """Cache immutable parse encodings, never mutable proposals or validation."""

    @lru_cache(maxsize=512)
    def encoded(key: _Key) -> tuple[str, str] | None:
        call = function(key)
        return (
            None if call is None else (call.tool_name, json.dumps(call.arguments, sort_keys=True))
        )

    @wraps(function)
    def fresh(key: _Key) -> ToolCall | None:
        result = encoded(key)
        return (
            None
            if result is None
            else ToolCall(tool_name=result[0], arguments=json.loads(result[1]))
        )

    return fresh


@lru_cache(maxsize=512)
def core_text(instruction: str) -> str:
    if len(instruction) > 10000:
        raise ValueError("instruction limit")
    text = instruction.strip()
    policies = (
        r"Work only inside the current (?:task workspace|directory)\.",
        r"Write ONLY the final answer to answer\.txt next to task\.json, (?:with no|without) explanation\.",
        r"(?:Compute(?: it)? using|Compute(?: it)? with|Use) python3(?: (?:for the calculation|to compute (?:the answer|an exact result from the available data)))?\.",
        r"(?:Return|Report|Supply|Give) (?:only )?(?:(?:the|its) )?(?:exact |final )?(?:integer(?: result| total)?|numerical value|result)(?: as one integer| only)?\.",
        r"Return their exact sum as one integer\.",
    )
    while True:
        for policy in policies:
            match = re.search(r"(?:^|\s)" + policy + r"$", text, re.I)
            if match:
                text = text[: match.start()].strip()
                break
        else:
            return text


@cached_call
def stock_call(instruction: str) -> ToolCall | None:
    text = core_text(instruction)
    start = re.match(
        r"(?:A |An )?(?:warehouse|depot|inventory) (?:starts with|begins with|has) (\d{1,12}) (boxes|cartons)\b",
        text,
        re.I,
    )
    if not start:
        return None
    body = text[start.end() :]
    unit = re.escape(start[2])
    question = rf"(?:How many {unit} (?:remain|are left)|Find its final number of {unit}(?: with python3)?|Determine its final stock using python3)[?.]$"
    ending = re.search(question, body, re.I)
    if not ending:
        return None
    body = body[: ending.start()]
    operations: list[JsonValue] = []
    patterns = (
        (rf"ships (\d+) {unit} (?:per day|each day) for (\d+) days", "ship"),
        (rf"receives (\d+) deliveries of (\d+) {unit}(?: each)?", "delivery"),
        (rf"receives (?:another |an extra )?(\d+) (?:new )?{unit}", "add"),
        (rf"discards (\d+) (?:damaged )?{unit}", "sub"),
        (rf"sends (?:out )?(\d+) batches of (\d+) {unit}", "ship"),
        (
            r"(?:is )?splits? (?:equally )?(?:among|into) (\d+) (?:warehouses|sites)(?: and it keeps one share \(round(?:ed)? down\))?",
            "split",
        ),
        (
            r"divides what remains equally among (\d+) sites while keeping one share rounded down",
            "floor_div",
        ),
        (r"doubles(?: that remaining share by merging with another warehouse| again)?", "double"),
    )
    while body.strip(" .,;\n"):
        body = body.lstrip(" .,;\n")
        body = re.sub(r"^(?:(?:and|then)\s+|(?:It|Its inventory)\s+)+", "", body, flags=re.I)
        for pattern, kind in patterns:
            match = re.match(pattern, body, re.I)
            if not match:
                continue
            values = list(map(int, match.groups()))
            end = match.end()
            if kind == "split":
                # Every division must explicitly keep one floored share.
                rest = body[end:]
                if "keeps one share" not in match[0].lower():
                    keep = re.match(r" and keeps one share \(round(?:ed)? down\)", rest, re.I)
                    if not keep:
                        return None
                    end += keep.end()
                kind = "floor_div"
            if kind in ("ship", "delivery"):
                op, value = ("sub" if kind == "ship" else "add"), values[0] * values[1]
            elif kind == "double":
                op, value = "mul", 2
            else:
                op, value = kind, values[0]
            if op == "floor_div" and value == 0:
                return None
            operations.append({"op": op, "value": value})
            body = body[end:]
            break
        else:
            return None
        if len(operations) > 32:
            return None
    return ToolCall(
        tool_name="local_math", arguments={"initial": int(start[1]), "operations": operations}
    )


@cached_call
def csv_call(task: TaskSpec) -> ToolCall | None:
    """Recognize aggregate/predicate clauses with explicit residual coverage.

    Structural stop words cannot express negation, alternatives, thresholds or
    numbers. Those require their own consumed, represented clauses below.
    """
    text = core_text(task.instruction)
    paths = re.findall(r"(?:workspace/)?[A-Za-z0-9_/-]+\.csv", text)
    if len(set(paths)) != 1:
        return None
    path = paths[0].removeprefix("workspace/")
    if path not in {a.path.as_posix() for a in selected_sources(task)}:
        return None
    work = text

    def consume(pattern: str, *, original: bool = False) -> list[re.Match]:
        nonlocal work
        found = list(re.finditer(pattern, text if original else work, re.I))
        for match in reversed(found):
            work = work[: match.start()] + " " * (match.end() - match.start()) + work[match.end() :]
        return found

    consume(re.escape(paths[0]))
    kinds = set()
    for pattern, kind in (
        (r"\b(?:total |combined )?(?:units-times-price )?revenue\b", "product_sum"),
        (
            r"\bhow many (?:order )?rows\b|\bcount (?:the )?(?:rows|records)\b|\bnumber of records\b",
            "count_rows",
        ),
        (
            r"\bhow many units\b|\b(?:find the )?sum of units\b|\badd (?:up )?(?:the )?units\b|\binteger unit count\b",
            "sum",
        ),
    ):
        if consume(pattern):
            kinds.add(kind)
    if len(kinds) != 1:
        return None
    aggregate = next(iter(kinds))
    if aggregate == "count_rows":
        consume(r"\bcount (?:the )?orders\b")
    if aggregate == "product_sum" and not re.search(
        r"units (?:multiplied by|times|[\u00d7*]) price|units-times-price", text, re.I
    ):
        return None
    # Explicit aggregate clarifications are compatible only with their operation.
    explanations = {
        "product_sum": (
            r"\(units (?:multiplied by|times|[\u00d7*]) price\)",
            r"Revenue per order is units times price;",
            r"multiply units by price for each included row and total those products\.",
        ),
        "count_rows": (
            r"Count rows, not units\.",
            r"Do not (?:total the units; return the number of records|sum quantities, count the qualifying records only)\.",
        ),
        "sum": (r", not the prices,",),
    }
    for pattern in explanations[aggregate]:
        consume(pattern, original=True)
    consume(
        r"(?:has columns|lists) (?:order_id,)?(?:product, region, status, units and price|region,product,month,units,price)\."
    )
    filters: list[dict] = []
    exclusions = consume(
        r"(?:Exclude nonmatching rows|exclude every other product, region, and status|ignore all other records|Exclude cheaper orders)\.",
        original=True,
    )
    value = r"'?([A-Za-z][A-Za-z0-9_-]{0,39})'?"

    def predicate(column, op, literal):
        if any(p["column"] == column and p["op"] == op for p in filters):
            raise ValueError("ambiguous duplicate predicate")
        filters.append({"column": column, "op": op, "value": literal})

    try:
        for match in consume(
            r"\b(product|region|status) (?:is )?EITHER " + value + r" OR " + value
        ):
            choices = [match[2], match[3]]
            if len(set(choices)) != len(choices):
                return None
            predicate(match[1].lower(), "in", sorted(choices))
        for column in ("product", "region", "status"):
            for match in consume(r"\b" + column + r"\s+(?:(?:is|equals)\s+)?" + value):
                predicate(column, "eq", match[1])
        for match in consume(
            r"\b(paid|shipped|pending|cancelled|refunded)\s+" + value + r"\s+orders\b"
        ):
            predicate("status", "eq", match[1])
            predicate("product", "eq", match[2])
        if not any(p["column"] == "product" for p in filters):
            for match in consume(r"\b" + value + r"\s+orders\b"):
                predicate("product", "eq", match[1])
        if not any(p["column"] == "product" for p in filters):
            for match in consume(r"\bfor\s+" + value):
                predicate("product", "eq", match[1])
        if not any(p["column"] == "region" for p in filters):
            for match in consume(r"\bin the\s+" + value + r"\s+region\b|\bin\s+" + value):
                literal = match[1] or match[2]
                if literal.casefold() in ("months", "month", "region"):
                    return None
                predicate("region", "eq", literal)
        ranges = consume(
            r"(?:\bfrom |\bbetween )?\bmonths? (?:must fall between )?(\d{1,2})\s*(?:through(?: month)?|and|[\u2013-])\s*(\d{1,2}),? inclusive"
        )
        for match in ranges:
            if not 1 <= int(match[1]) <= int(match[2]) <= 12:
                return None
            predicate("month", "ge", str(int(match[1])))
            predicate("month", "le", str(int(match[2])))
        for match in consume(r"\bunit price is at least (\d{1,9}(?:\.\d{1,4})?)\b"):
            predicate("price", "ge", match[1])
    except ValueError:
        return None
    if not filters:
        return None
    for sentence in exclusions:
        if "cheaper" in sentence[0].casefold() and not any(
            p["column"] == "price" and p["op"] == "ge" for p in filters
        ):
            return None
        if "every other" in sentence[0].casefold() and not {"product", "region", "status"}.issubset(
            {p["column"] for p in filters}
        ):
            return None
    for match in consume(r"Exclude ([A-Za-z, ]+)\."):
        excluded = re.split(r",\s*(?:and\s+)?|\s+and\s+", match[1])
        status = [p for p in filters if p["column"] == "status"]
        if len(status) != 1 or status[0]["op"] not in ("eq", "in"):
            return None
        included = [status[0]["value"]] if status[0]["op"] == "eq" else status[0]["value"]
        if set(excluded) & set(included):
            return None
    consume(
        r"(?:Exclude nonmatching rows|exclude every other product, region, and status|ignore all other records|Exclude cheaper orders)\."
    )
    glue = set(
        "the csv at report in from using read use records of for with whose where have are is and their must fall during placed sold orders product region status calculate find total exact what how many were when but only return one integer answer not".split()
    )
    # `not` is deliberately not a structural token, despite common English glue.
    glue.discard("not")
    if any(token.casefold() not in glue for token in re.findall(r"[A-Za-z0-9_]+", work)):
        return None
    return ToolCall(
        tool_name="query_csv",
        arguments=cast(
            JsonObject,
            {
                "path": path,
                "aggregate": aggregate,
                "columns": []
                if aggregate == "count_rows"
                else ["units", "price"]
                if aggregate == "product_sum"
                else ["units"],
                "filters": filters,
            },
        ),
    )


@cached_call
def search_call(task: TaskSpec) -> ToolCall | None:
    """Profile relation composition with explicit vocabulary/residual coverage."""
    text = core_text(task.instruction)
    paths = [a.path.as_posix() for a in selected_sources(task) if a.path.suffix.casefold() == ".md"]
    if not 0 < len(paths) <= 128:
        return None
    names = []
    for match in re.finditer(r"[A-Z][a-z]+(?: [A-Z][a-z]+){1,3}", text):
        name = match[0]
        start = match.start()
        if (
            len(name.split()) >= 3
            and (start == 0 or text[:start].rstrip().endswith((".", "?")))
            and name.split()[0].casefold()
            in {
                "find",
                "locate",
                "follow",
                "identify",
                "consult",
                "report",
            }
        ):
            prefix, _, name = name.partition(" ")
            start += len(prefix) + 1
        if simple_entity(name):
            names.append((name, start))
    if len(names) != 1:
        return None
    entity, start = names[0]
    work = text[:start] + " " * len(entity) + text[start + len(entity) :]
    work = re.sub(r"(?:workspace/)?[A-Za-z0-9_/-]+/", " ", work)
    keys = (
        (r"research summit coordinator", "Research summit coordinator"),
        (r"program coordinator", "Program coordinator"),
        (r"parent organization", "Parent organization"),
        (r"employer|employed by a company", "Employer"),
        (r"mentors?|mentor", "Mentor"),
        (r"lab", "Lab"),
    )
    relations = []
    for pattern, key in keys:
        for match in re.finditer(r"\b" + pattern + r"\b", work, re.I):
            relations.append((match.start(), key))
    chain = list(dict.fromkeys(key for _, key in sorted(relations)))
    years = re.findall(r"\b\d{4}\b", work)
    field = None
    initiative = re.search(r"(?:the )?([A-Z][A-Za-z]*(?: [A-Z][A-Za-z]*)*) initiative", work)
    if re.search(r"\bborn|birth year", work, re.I):
        field = "Born"
    elif re.search(r"\bfounded|founding year", work, re.I):
        field = "Founded"
    elif years and re.search(r"\bsummit\b", work, re.I):
        if len(years) != 1:
            return None
        field = years[0] + " summit host city"
        work = work.replace(years[0], " ")
    elif initiative and re.search(r"\bmonth\b", work, re.I):
        field = initiative[1] + " initiative launch month"
        work = work[: initiative.start()] + " " * len(initiative[0]) + work[initiative.end() :]
    elif re.search(r"\bheadquarter(?:s|ed)?\b", work, re.I):
        field = "Headquarters"
    if field is None or len(chain) > 4:
        return None
    if (
        sum(
            bool(re.search(pattern, work, re.I))
            for pattern in (
                r"\bborn|birth year",
                r"\bfounded|founding year",
                r"\bheadquarter(?:s|ed)?\b",
                r"\bmonth\b",
                r"\b\d{4} summit\b",
            )
        )
        > 1
    ):
        return None
    if len(chain) > 1 and not re.search(
        r"then|\bits\b|whose|that|(?:mentor|employer|parent|coordinator)'s|follow", work, re.I
    ):
        return None
    # Explicit disambiguation can exclude the intermediate employer, never a
    # missing/negated link or an arbitrary extra condition.
    if re.search(r"not the employer's", work, re.I):
        if not chain or chain[-1] != "Parent organization":
            return None
        work = re.sub(r"not the employer's", " ", work, flags=re.I)
    for pattern, _ in keys:
        work = re.sub(r"\b" + pattern + r"\b", " ", work, flags=re.I)
    glue = set(
        "consult the markdown profiles in what year was born founded use only those records directory under workspace contains person find birth give four digit from local kb locate and return city listed for headquarters identify of which is that located employed by a an company whose appears follow both links employer headquartered who then report its employer refer first research summit organization named as hosted coordinator s host english month did launch associated with their to founding parent final provided documents link connects complete chain program initiative using belongs".split()
    )
    if any(token.casefold() not in glue for token in re.findall(r"[A-Za-z0-9_]+", work)):
        return None
    return ToolCall(
        tool_name="lookup_markdown",
        arguments=cast(
            JsonObject, {"paths": paths, "entity": entity, "chain": chain, "field": field}
        ),
    )


def simple_search(task: TaskSpec) -> ToolCall | None:
    call = search_call(task)
    args = cast(dict, call.arguments) if call else {}
    if call and len(args["chain"]) <= 1 and args["field"] in ("Born", "Headquarters"):
        return call
    return None


def simple_csv(task: TaskSpec) -> ToolCall | None:
    call = csv_call(task)
    args = cast(dict, call.arguments) if call else {}
    if (
        call
        and args["aggregate"] in ("sum", "count_rows")
        and all(
            p["column"] in ("product", "region", "month") and p["op"] in ("eq", "ge", "le")
            for p in args["filters"]
        )
    ):
        return call
    return None


@lru_cache(maxsize=512)
def discrete_spec(instruction: str) -> tuple[str, tuple[int, ...]] | None:
    text = core_text(instruction)
    prime = re.fullmatch(
        r"(?:Add together|(?:Find|Calculate|What is)(?: the)?(?: combined)? (?:sum|total)(?: of)?)(?: the)? (?:first|earliest) (\d{1,4}) (?:prime numbers|primes)(?: in ascending order)?(?:, (?:starting with|counting) 2(?: as the first prime)?| including 2| \(begin at 2\))?[.?]",
        text,
        re.I,
    )
    if prime and 1 <= int(prime[1]) <= 1000:
        return "prime_sum", (int(prime[1]),)
    for pattern in (
        r"(?:Find|What is) the smallest positive integer (?:that is )?divisible by (?:each of )?([\d, and]+) (?:without a remainder|with no remainder)[.?]",
        r"(?:Find|Calculate) the least common positive multiple (?:shared by|of) ([\d, and]+)[.?]",
        r"Multiple (?:(?:inspection|maintenance|production|service) )?cycles repeat every ([\d, and]+) days\. What is the first positive day count on which all cycles line up again\?",
    ):
        match = re.fullmatch(pattern, text, re.I)
        if match:
            pieces = re.split(r",\s*(?:and\s+)?|\s+and\s+", match[1])
            if 2 <= len(pieces) <= 16 and all(
                re.fullmatch(r"\d{1,6}", p) and int(p) > 0 for p in pieces
            ):
                return "lcm", tuple(map(int, pieces))
    rows = re.search(r"(\d{1,6}) rows", text, re.I)
    seats = list(re.finditer(r"(\d{1,6}) (?:more |additional )?seats", text, re.I))
    if len(seats) == 1:
        seats += list(re.finditer(r"(\d{1,6}) more than", text, re.I))
    numbers = re.findall(r"\d+", text)
    if (
        rows
        and len(seats) == 2
        and len(numbers) == 3
        and re.search(r"(?:each|every|following|grows)", text, re.I)
    ):
        remaining = re.sub(r"\d+", " ", text)
        glue = set(
            "a an small lecture hall auditorium theatre theater tiered venue has consists of rows seats the first in row one starting at every next each subsequent more than previous preceding and size grows by time additional following count all total available altogether how many are there compute capacity with".split()
        )
        if all(w.casefold() in glue for w in re.findall(r"[A-Za-z]+", remaining)):
            values = (int(rows[1]), int(seats[0][1]), int(seats[1][1]))
            if values[0] > 0:
                return "series", values
    committee = re.fullmatch(
        r"A (?:(?:[A-Za-z]+ )?club|group|team|organization) must choose a committee of (\d{1,3}) different people from (\d{1,4}) volunteers\. The order in which people are chosen does not matter\. How many distinct committees can be formed\?",
        text,
        re.I,
    )
    if (
        committee
        and 1 <= int(committee[1]) <= min(20, int(committee[2]))
        and int(committee[2]) <= 1000
    ):
        return "combinations", (int(committee[2]), int(committee[1]))
    codes = re.fullmatch(
        r"A (?:security access|lock) code (?:uses|has) exactly (\d{1,2}) (?:distinct|different) digits chosen from ([\d, and]+)\. (?:Position|The order) matters,? and (?:the|a) code (?:IS )?allowed to start with 0\. How many (?:different codes exist|distinct codes are possible)\?",
        text,
        re.I,
    )
    if codes:
        digits = re.split(r",\s*(?:and\s+)?|\s+and\s+", codes[2])
        if (
            all(re.fullmatch(r"\d", d) for d in digits)
            and len(set(digits)) == len(digits)
            and 1 <= int(codes[1]) <= len(digits) <= 10
        ):
            return "arrangements", (len(digits), int(codes[1]))
    multiples = re.fullmatch(
        r"Consider all positive integers no greater than (\d{1,6}) that are divisible by (\d{1,6})\. What is their sum\? Include (\d{1,6}) if divisible, and answer as an integer\.",
        text,
        re.I,
    )
    if (
        multiples
        and multiples[1] == multiples[3]
        and 1 <= int(multiples[2]) <= int(multiples[1]) <= 100000
    ):
        return "multiples", (int(multiples[1]), int(multiples[2]))
    return None
