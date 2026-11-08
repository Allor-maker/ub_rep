"""Closed instruction semantics and independent local checks for data plans.

The model proposes a ToolCall. This module binds it to the entire instruction;
successful execution alone never authorizes a candidate. No data is sent to LLMs.
"""

import csv
import hashlib
import io
import json
import re
import time
from codecs import BOM_UTF8
from dataclasses import dataclass
from decimal import Context, Decimal, Inexact, localcontext
from pathlib import PurePosixPath
from uuid import uuid4

from nsu_budget_agent.contracts.models import (
    ArtifactRef,
    TaskSpec,
    ToolCall,
    ToolContext,
    ToolLimits,
    parse_rel_path,
)
from nsu_budget_agent.contracts.ports import ToolExecutor
from nsu_budget_agent.task_io._files import excluded

_FILE = r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*\.csv"
_DIRECTORY = r"[A-Za-z0-9_-]+(?:/[A-Za-z0-9_-]+)*"
_COLUMN = r"[A-Za-z_][A-Za-z0-9_]{0,39}"
_VALUE = r"[^'\r\n]{1,80}"
_ENTITY = r"[A-Za-z][A-Za-z .'-]{0,79}?"
_AMBIGUOUS_WORDS = frozenset(
    "and or nor not but except unless without excluding only either neither with who whose "
    "which where if then rather than including of from in at for by is are was were "
    "born based located living working before after since until while no never else instead".split()
)


@dataclass(frozen=True)
class DataQuestion:
    call: ToolCall
    paths: tuple[PurePosixPath, ...]


class EvidenceUnavailable(RuntimeError):
    """Verification could not read trusted evidence; this is inconclusive."""


def _unambiguous_entity(entity: str) -> bool:
    # Unquoted free text can otherwise absorb "Alice or Bob" or "Alice from
    # London" as a single name. This deliberately rejects some valid names.
    words = re.findall(r"[A-Za-z]+", entity)
    return (
        bool(words)
        and re.search(r"(?<!\b[A-Z])\.", entity) is None
        and re.search(r"'s\b", entity, re.IGNORECASE) is None
        and all(word[0].isupper() and word.casefold() not in _AMBIGUOUS_WORDS for word in words)
    )


def evidence_fits(task: TaskSpec, spec: DataQuestion, limits: ToolLimits) -> bool:
    """Reject only sizes that cannot pass existing read and validation limits."""
    allowed = {artifact.path: artifact for artifact in task.artifacts}
    if any(path not in allowed for path in spec.paths):
        return False
    if sum(artifact.size_bytes for artifact in task.artifacts) > min(
        limits.max_read_bytes, 8 * 1024 * 1024
    ):
        return False
    for path in spec.paths:
        # read_file decodes utf-8-sig: at most one three-byte BOM disappears.
        # JSON escaping can increase the size but cannot reduce this lower bound.
        minimum_text_bytes = max(0, allowed[path].size_bytes - len(BOM_UTF8))
        empty_payload = json.dumps(
            {"text": "", "path": path.as_posix()}, ensure_ascii=False
        ).encode("utf-8")
        if minimum_text_bytes + len(empty_payload) > limits.max_output_bytes:
            return False
    return True


def _matches_original_text(text: str, artifact: ArtifactRef) -> bool:
    """Reconstruct the only UTF-8/UTF-8-BOM inputs that decode to this text."""
    if len(text) > artifact.size_bytes:
        return False
    try:
        encoded = text.encode("utf-8")
    except UnicodeEncodeError:
        return False
    for raw in (encoded, BOM_UTF8 + encoded):
        if (
            len(raw) == artifact.size_bytes
            and raw.decode("utf-8-sig") == text
            and hashlib.sha256(raw).hexdigest() == artifact.sha256
        ):
            return True
    return False


def data_question(task: TaskSpec) -> DataQuestion | None:
    """Fully anchored, explicit columns/entities/relations; no inferred intent."""
    if (
        task.type_hint == "coding"
        or task.check_hint not in (None, "exact_str", "exact_int")
        or len(task.instruction) > 2048
    ):
        return None
    text = task.instruction.strip()
    csv_patterns = (
        (
            rf"From (?P<path>{_FILE}), total the (?P<c1>{_COLUMN}) values for rows whose (?P<f>{_COLUMN}) is '(?P<v>{_VALUE})'\.",
            "sum",
        ),
        (
            rf"Count the records in (?P<path>{_FILE}) with (?P<f>{_COLUMN}) equal to '(?P<v>{_VALUE})'\.",
            "count_rows",
        ),
        (
            rf"Using (?P<path>{_FILE}), calculate the sum of (?P<c1>{_COLUMN}) \* (?P<c2>{_COLUMN}) for entries with (?P<f>{_COLUMN}) '(?P<v>{_VALUE})'\.",
            "product_sum",
        ),
    )
    for pattern, aggregate in csv_patterns:
        match = re.fullmatch(pattern, text)
        if match:
            path = parse_rel_path(match["path"])
            if (
                path not in {a.path for a in task.artifacts}
                or any(excluded(part) for part in path.parts)
                or not match["v"].strip()
            ):
                return None
            columns = [match.groupdict()[c] for c in ("c1", "c2") if c in match.groupdict()]
            return DataQuestion(
                ToolCall(
                    tool_name="query_csv",
                    arguments={
                        "path": path.as_posix(),
                        "aggregate": aggregate,
                        "columns": columns,
                        "filters": [{"column": match["f"], "op": "eq", "value": match["v"]}],
                    },
                ),
                (path,),
            )
    searches = (
        (
            rf"Look up the (?P<field>Email|Department) of (?P<entity>{_ENTITY}) in (?P<directory>{_DIRECTORY})/\*\.md\.",
            (),
        ),
        (
            rf"Which city is listed as (?P<field>Headquarters) for the Employer of (?P<entity>{_ENTITY}) in (?P<directory>{_DIRECTORY})/\*\.md\?",
            ("Employer",),
        ),
        (
            rf"According to (?P<directory>{_DIRECTORY})/\*\.md, give the (?P<field>Born|Founded) year for (?P<entity>{_ENTITY})\.",
            (),
        ),
    )
    for pattern, chain in searches:
        match = re.fullmatch(pattern, text)
        if match:
            if not _unambiguous_entity(match["entity"]):
                return None
            directory = parse_rel_path(match["directory"])
            if any(excluded(part) for part in directory.parts):
                return None
            # *.md means immediate children, not recursive or task-global search.
            paths = tuple(
                sorted(
                    a.path
                    for a in task.artifacts
                    if a.path.parent == directory
                    and a.path.suffix == ".md"
                    and not any(excluded(part) for part in a.path.parts)
                )
            )
            if not 0 < len(paths) <= 32:
                return None
            return DataQuestion(
                ToolCall(
                    tool_name="lookup_markdown",
                    arguments={
                        "paths": [p.as_posix() for p in paths],
                        "entity": match["entity"],
                        "chain": list(chain),
                        "field": match["field"],
                    },
                ),
                paths,
            )
    return None


def matches_instruction(task: TaskSpec, call: ToolCall) -> bool:
    spec = data_question(task)
    return spec is not None and spec.call == call


def _csv_answer(text: str, args: dict, deadline: float) -> str:
    # Independent of tools.calculations/csv_audit: Decimal + DictReader, not
    # the executor's Fraction aggregation. Reject blanks even in filtered rows.
    reader = csv.DictReader(io.StringIO(text, newline=""), strict=True)
    fields = reader.fieldnames
    if (
        not fields
        or len(fields) > 128
        or any(not c for c in fields)
        or len(fields) != len(set(fields))
    ):
        raise ValueError("invalid CSV header")
    predicate = args["filters"][0]
    required = [*args["columns"], predicate["column"]]
    if any(column not in fields for column in required):
        raise ValueError("missing column")
    context = Context(prec=256, Emin=-1024, Emax=1024)
    context.traps[Inexact] = True
    with localcontext(context):
        total = Decimal(0)
        for count, row in enumerate(reader, start=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("independent CSV deadline")
            if count > 100_000 or None in row or any(value is None for value in row.values()):
                raise ValueError("CSV shape/row limit")
            if not row[predicate["column"]].strip():
                raise ValueError("unknown filter value")
            numbers = []
            for column in args["columns"]:
                if re.fullmatch(r"[+-]?\d{1,40}(?:\.\d{1,20})?", row[column]) is None:
                    raise ValueError("invalid numeric cell")
                numbers.append(Decimal(row[column]))
            if row[predicate["column"]] == predicate["value"]:
                if args["aggregate"] == "count_rows":
                    total += 1
                elif args["aggregate"] == "sum":
                    total += numbers[0]
                else:
                    total += numbers[0] * numbers[1]
        rendered = format(total, "f")
        return (rendered.rstrip("0").rstrip(".") if "." in rendered else rendered) if total else "0"


def _markdown_answer(
    documents: dict[PurePosixPath, str], args: dict, deadline: float
) -> tuple[str, tuple[PurePosixPath, ...]]:
    # Separate parser/lookup from the executor; plain properties are data only.
    profiles: dict[str, tuple[dict[str, str], PurePosixPath]] = {}
    for path, document in documents.items():
        lines = [line.strip() for line in document.splitlines() if line.strip()]
        if not lines or not re.fullmatch(r"# [^#]+", lines[0]):
            raise ValueError("invalid profile heading")
        name = lines[0][2:].strip()
        if name in profiles:
            raise ValueError("duplicate entity")
        properties = {}
        for line in lines[1:]:
            if time.monotonic() >= deadline:
                raise TimeoutError("independent Markdown deadline")
            field, separator, value = line.partition(":")
            if (
                not separator
                or not re.fullmatch(r"[A-Za-z][A-Za-z ]*", field)
                or not value.strip()
                or field.casefold() in properties
            ):
                raise ValueError("ambiguous profile")
            properties[field.casefold()] = value.strip()
        profiles[name] = properties, path
    entity = args["entity"]
    evidence = set()
    for field in [*args["chain"], args["field"]]:
        if entity not in profiles:
            raise ValueError("missing entity")
        properties, path = profiles[entity]
        entity = properties[field.casefold()]
        evidence.add(path)
    if args["field"] in ("Born", "Founded") and not re.fullmatch(r"\d{1,4}", entity):
        raise ValueError("invalid year")
    if args["field"] == "Email" and not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", entity):
        raise ValueError("invalid email")
    return entity, tuple(sorted(evidence))


async def independently_answer(
    task: TaskSpec, tools: ToolExecutor, ctx: ToolContext
) -> tuple[str, tuple[PurePosixPath, ...]]:
    spec = data_question(task)
    if spec is None:
        raise ValueError("no supported semantic interpretation")
    allowed = {artifact.path: artifact for artifact in ctx.source_artifacts}
    original = {artifact.path: artifact for artifact in task.artifacts}
    if any(path not in allowed for path in spec.paths) or sum(
        allowed[path].size_bytes for path in spec.paths
    ) > min(ctx.limits.max_read_bytes, 8 * 1024 * 1024):
        raise ValueError("independent evidence limit")
    documents: dict[PurePosixPath, str] = {}
    for path in spec.paths:
        call_id = uuid4().hex
        checked = await tools.execute(
            call_id, ToolCall(tool_name="read_file", arguments={"path": path.as_posix()}), ctx
        )
        if (
            checked.call_id != call_id
            or checked.tool_name != "read_file"
            or checked.status != "ok"
            or checked.truncated
            or not isinstance(checked.data, dict)
        ):
            raise EvidenceUnavailable("independent source read failed")
        data = checked.data
        text = data.get("text")
        if data.get("path") != path.as_posix() or not isinstance(text, str):
            raise EvidenceUnavailable("invalid independent source")
        if (
            path not in original
            or (allowed[path].size_bytes, allowed[path].sha256)
            != (original[path].size_bytes, original[path].sha256)
            or not _matches_original_text(text, original[path])
        ):
            raise ValueError("independent source does not match original artifact")
        documents[path] = text
    if spec.call.tool_name == "query_csv":
        return _csv_answer(
            documents[spec.paths[0]], spec.call.arguments, ctx.deadline_monotonic
        ), spec.paths
    return _markdown_answer(documents, spec.call.arguments, ctx.deadline_monotonic)
