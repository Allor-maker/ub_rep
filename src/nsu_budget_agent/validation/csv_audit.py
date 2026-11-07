"""Independent CSV completeness and order/row counts; no solver aggregation imports."""

import csv
import io
import time
from typing import Any


def audit_csv(text: str, arguments: dict[str, Any], deadline: float) -> tuple[bool, str | None]:
    reader = csv.reader(io.StringIO(text, newline=""), strict=True)
    header = next(reader, [])
    if not header or any(not name for name in header) or len(header) != len(set(header)):
        return False, None
    filters = arguments.get("filters")
    if not isinstance(filters, list):
        return False, None
    filter_columns = []
    for predicate in filters:
        if (
            not isinstance(predicate, dict)
            or set(predicate) != {"column", "op", "value"}
            or predicate["column"] not in header
            or not isinstance(predicate["value"], str)
            or not predicate["value"].strip()
        ):
            return False, None
        filter_columns.append(predicate["column"])
    aggregate = arguments.get("aggregate")
    counting = aggregate in ("count", "count_rows", "count_distinct")
    if counting and any(predicate["op"] != "eq" for predicate in filters):
        return False, None
    declared = arguments.get("expected_columns")
    if declared is not None and (
        not isinstance(declared, list)
        or len(declared) != len(set(declared))
        or set(declared) != set(header)
    ):
        return False, None
    if aggregate == "count_distinct" and (
        arguments.get("columns") != ["order_id"]
        or not isinstance(declared, list)
        or "order_id" not in declared
        or "order_id" not in header
    ):
        return False, None
    ids: set[str] = set()
    case_ids: dict[str, str] = {}
    regions: dict[str, str] = {}
    matched_rows = 0
    rows = 0
    for values in reader:
        if time.monotonic() >= deadline:
            raise TimeoutError("independent CSV audit deadline")
        if not values:
            continue
        rows += 1
        if rows > 100_000 or len(values) != len(header):
            return False, None
        row = dict(zip(header, values, strict=True))
        if any(not row[column].strip() for column in filter_columns):
            return False, None
        if aggregate == "count_distinct":
            order_id = row["order_id"]
            if not order_id.strip() or order_id != order_id.strip():
                return False, None
            key = order_id.casefold()
            if key in case_ids and case_ids[key] != order_id:
                return False, None
            case_ids[key] = order_id
            if "region" in filter_columns:
                if order_id in regions and regions[order_id] != row["region"]:
                    return False, None
                regions[order_id] = row["region"]
        if counting and all(row[p["column"]] == p["value"] for p in filters):
            matched_rows += 1
            if aggregate == "count_distinct":
                ids.add(row["order_id"])
    expected = arguments.get("expected_rows")
    if time.monotonic() >= deadline:
        raise TimeoutError("independent CSV audit deadline")
    if expected is not None and (type(expected) is not int or expected != rows):
        return False, None
    if aggregate == "count_distinct":
        return True, str(len(ids))
    if aggregate in ("count", "count_rows"):
        return True, str(matched_rows)
    return True, None
