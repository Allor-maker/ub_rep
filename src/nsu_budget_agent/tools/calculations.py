"""Bounded exact arithmetic and strict CSV/Markdown interpretations."""

import ast
import csv
import io
import re
from fractions import Fraction
from typing import Callable


def number(value: str) -> Fraction:
    if not re.fullmatch(r"[+-]?\d{1,40}(?:\.\d{1,20})?", value):
        raise ValueError("invalid bounded decimal")
    return bounded(Fraction(value))


def bounded(value: Fraction) -> Fraction:
    if max(value.numerator.bit_length(), value.denominator.bit_length()) > 512:
        raise ValueError("arithmetic size limit")
    return value


def render(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    for factor in (2, 5):
        while denominator % factor == 0:
            denominator //= factor
    if denominator != 1:
        raise ValueError("non-terminating result has no specified output representation")
    sign = "-" if value < 0 else ""
    value = abs(value)
    whole, remainder = divmod(value.numerator, value.denominator)
    digits = ""
    while remainder and len(digits) < 40:
        digit, remainder = divmod(remainder * 10, value.denominator)
        digits += str(digit)
    if remainder:
        raise ValueError("decimal output limit")
    return f"{sign}{whole}.{digits}"


def arithmetic(arguments: dict, check: Callable[[], None]) -> str:
    if set(arguments) == {"expression"}:
        expression = arguments["expression"]
        if not isinstance(expression, str) or not 0 < len(expression) <= 1024:
            raise ValueError("invalid expression")
        tree = ast.parse(expression, mode="eval")
        if sum(1 for _ in ast.walk(tree)) > 128:
            raise ValueError("expression node limit")

        def evaluate(node: ast.AST, depth: int = 0) -> Fraction:
            check()
            if depth > 16:
                raise ValueError("expression depth limit")
            if isinstance(node, ast.Constant) and type(node.value) in (int, float):
                # Recover the decimal lexeme; never account through binary float.
                return number(ast.get_source_segment(expression, node) or "")
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
                value = evaluate(node.operand, depth + 1)
                return -value if isinstance(node.op, ast.USub) else value
            if isinstance(node, ast.BinOp):
                left, right = evaluate(node.left, depth + 1), evaluate(node.right, depth + 1)
                if isinstance(node.op, ast.Add):
                    return bounded(left + right)
                if isinstance(node.op, ast.Sub):
                    return bounded(left - right)
                if isinstance(node.op, ast.Mult):
                    return bounded(left * right)
                if isinstance(node.op, ast.Div):
                    return bounded(left / right)
                if isinstance(node.op, ast.FloorDiv):
                    return bounded(Fraction(left // right))
                if isinstance(node.op, ast.Mod):
                    return bounded(left % right)
                if isinstance(node.op, ast.Pow) and right.denominator == 1 and 0 <= right <= 12:
                    return bounded(left ** int(right))
            raise ValueError("expression operation unavailable")

        return render(evaluate(tree.body))
    if set(arguments) != {"initial", "operations"} or type(arguments["initial"]) is not int:
        raise ValueError("invalid arithmetic arguments")
    value = arguments["initial"]
    operations = arguments["operations"]
    if (
        value < 0
        or value.bit_length() > 512
        or not isinstance(operations, list)
        or len(operations) > 64
    ):
        raise ValueError("invalid stock operations")
    for operation in operations:
        check()
        if not isinstance(operation, dict) or set(operation) != {"op", "value"}:
            raise ValueError("invalid operation")
        operand = operation["value"]
        if type(operand) is not int or operand < 0 or operand.bit_length() > 256:
            raise ValueError("invalid operand")
        match operation["op"]:
            case "add":
                value += operand
            case "sub":
                value -= operand
            case "mul":
                value *= operand
            case "floor_div":
                value //= operand
            case _:
                raise ValueError("unknown operation")
        if value < 0 or value.bit_length() > 512:
            raise ValueError("impossible or oversized stock")
    return str(value)


def csv_query(text: str, arguments: dict, check: Callable[[], None]) -> str:
    if not {"path", "aggregate", "columns", "filters"} <= arguments.keys() or set(arguments) - {
        "path",
        "aggregate",
        "columns",
        "filters",
        "expected_rows",
        "expected_columns",
    }:
        raise ValueError("invalid CSV arguments")
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, [])
    if (
        not header
        or any(not name for name in header)
        or len(set(header)) != len(header)
        or len(header) > 128
    ):
        raise ValueError("invalid or duplicate CSV columns")
    if "expected_columns" in arguments:
        declared = arguments["expected_columns"]
        if (
            not isinstance(declared, list)
            or any(not isinstance(c, str) for c in declared)
            or len(declared) != len(set(declared))
            or set(header) != set(declared)
        ):
            raise ValueError("CSV schema differs from instruction")
    columns, filters = arguments["columns"], arguments["filters"]
    aggregate = arguments["aggregate"]
    if (
        aggregate not in ("count", "count_rows", "count_distinct", "sum", "product_sum")
        or not isinstance(columns, list)
        or not isinstance(filters, list)
        or len(filters) > 16
    ):
        raise ValueError("invalid aggregate")
    if len(columns) != {
        "count": 0,
        "count_rows": 0,
        "count_distinct": 1,
        "sum": 1,
        "product_sum": 2,
    }[aggregate] or any(c not in header for c in columns):
        raise ValueError("missing aggregate columns")
    if aggregate == "count_distinct" and (
        columns != ["order_id"]
        or "expected_columns" not in arguments
        or "order_id" not in arguments["expected_columns"]
    ):
        raise ValueError("order identity is not established by the declared schema")
    for predicate in filters:
        if not isinstance(predicate, dict) or set(predicate) != {"column", "op", "value"}:
            raise ValueError("invalid predicate")
        if (
            predicate["column"] not in header
            or predicate["op"] not in ("eq", "ge", "le")
            or not isinstance(predicate["value"], str)
            or not predicate["value"].strip()
        ):
            raise ValueError("invalid filter")
    total, count = Fraction(0), 0
    orders: set[str] = set()
    identity_cases: dict[str, str] = {}
    order_regions: dict[str, str] = {}
    for values in reader:
        check()
        if not values:  # CSV files may contain blank physical records.
            continue
        count += 1
        if count > 100_000 or len(values) != len(header):
            raise ValueError("CSV row/shape limit")
        row = dict(zip(header, values, strict=True))
        for predicate in filters:
            if not row[predicate["column"]].strip():
                raise ValueError("missing required filter value")
        if aggregate == "count_distinct":
            order_id = row["order_id"]
            if not order_id.strip() or order_id != order_id.strip():
                raise ValueError("missing or ambiguous order_id")
            key = order_id.casefold()
            if key in identity_cases and identity_cases[key] != order_id:
                raise ValueError("ambiguous case variants of order_id")
            identity_cases[key] = order_id
            if any(p["column"] == "region" for p in filters):
                region = row["region"]
                if order_id in order_regions and order_regions[order_id] != region:
                    raise ValueError("conflicting regions for one order")
                order_regions[order_id] = region
        accepted = True
        for predicate in filters:
            left, right = row[predicate["column"]], predicate["value"]
            if predicate["op"] == "eq":
                accepted &= left == right
            elif predicate["op"] == "ge":
                accepted &= number(left) >= number(right)
            else:
                accepted &= number(left) <= number(right)
        # Check every numeric aggregate cell, even if its row is filtered out.
        value = Fraction(1)
        if aggregate != "count_distinct":
            for column in columns:
                value = bounded(value * number(row[column]))
        if accepted:
            if aggregate == "count_distinct":
                orders.add(row["order_id"])
            else:
                total = bounded(total + value)
    expected = arguments.get("expected_rows")
    if expected is not None and (type(expected) is not int or expected != count):
        raise ValueError("CSV row count differs from instruction")
    return str(len(orders)) if aggregate == "count_distinct" else render(total)


def markdown_lookup(
    documents: dict[str, str], arguments: dict, check: Callable[[], None]
) -> tuple[str, list[str]]:
    if set(arguments) != {"paths", "entity", "chain", "field"}:
        raise ValueError("invalid lookup arguments")
    entity, chain, field = arguments["entity"], arguments["chain"], arguments["field"]
    if (
        not isinstance(entity, str)
        or not isinstance(chain, list)
        or len(chain) > 4
        or not all(isinstance(c, str) for c in chain)
        or not isinstance(field, str)
    ):
        raise ValueError("invalid lookup path")
    profiles = {}
    for path, text in documents.items():
        check()
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines or not re.fullmatch(r"# [^#]+", lines[0]):
            raise ValueError("Markdown profile requires one entity heading")
        name = lines[0][2:].strip()
        properties = {}
        for line in lines[1:]:
            match = re.fullmatch(r"([A-Za-z][A-Za-z ]*):\s*(\S.*)", line)
            if not match or match[1].casefold() in properties:
                raise ValueError("ambiguous or unstructured Markdown profile")
            properties[match[1].casefold()] = match[2].strip()
        if name in profiles:
            raise ValueError("duplicate entity identity")
        profiles[name] = (properties, path)
    evidence = []
    for relation in [*chain, field]:
        check()
        if entity not in profiles:
            raise ValueError("entity evidence missing")
        properties, path = profiles[entity]
        if relation.casefold() not in properties:
            raise ValueError("requested fact missing")
        evidence.append(path)
        entity = properties[relation.casefold()]
    if field.casefold() in ("born", "founded") and not re.fullmatch(r"\d{1,4}", entity):
        raise ValueError("invalid explicit year")
    return entity, sorted(set(evidence))
