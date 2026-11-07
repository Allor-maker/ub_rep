"""Bounded, read-only staging tools. No task code, imports or shell execution."""

import csv
import hashlib
import json
import os
import time
from pathlib import PurePosixPath

from nsu_budget_agent.contracts.models import (
    ErrorInfo,
    Id,
    OperationFailed,
    ToolCall,
    ToolContext,
    ToolResult,
    ToolSpec,
    parse_rel_path,
)
from nsu_budget_agent.task_io._files import open_directory
from nsu_budget_agent.task_io.workspace import artifact_file, joined_io

from .calculations import arithmetic, csv_query, markdown_lookup


class LocalToolExecutor:
    def catalog(self) -> tuple[ToolSpec, ...]:
        return tuple(
            ToolSpec(
                name=name,
                description=description,
                input_schema={"type": "object"},
                output_schema={"type": "object"},
                mutates_files=False,
            )
            for name, description in (
                ("read_file", "Read one allowed UTF-8 text, Markdown or CSV artifact"),
                ("local_math", "Bounded exact arithmetic or integer stock operations"),
                (
                    "query_csv",
                    "Exact CSV sum, product-sum, row count or distinct order count with explicit filters",
                ),
                ("audit_csv", "Independently audit CSV completeness and count identities or rows"),
                ("lookup_markdown", "Resolve explicit entity attributes through Markdown profiles"),
            )
        )

    async def execute(self, call_id: Id, call: ToolCall, ctx: ToolContext) -> ToolResult:
        started = time.monotonic()
        try:
            data = await joined_io(self._execute, call, ctx)
            if (
                len(json.dumps(data, ensure_ascii=False).encode("utf-8"))
                > ctx.limits.max_output_bytes
            ):
                raise OperationFailed(ErrorInfo(code="resource_limit", message="tool output limit"))
            return ToolResult(
                call_id=call_id,
                tool_name=call.tool_name,
                status="ok",
                data=data,
                duration_s=time.monotonic() - started,
            )
        except OperationFailed as exc:
            error = exc.error
        except (
            ValueError,
            TypeError,
            KeyError,
            OSError,
            OverflowError,
            ZeroDivisionError,
            SyntaxError,
            RecursionError,
            csv.Error,
        ):
            error = ErrorInfo(
                code="invalid_input", message="unsupported, incomplete or malformed local evidence"
            )
        return ToolResult(
            call_id=call_id,
            tool_name=call.tool_name,
            status="error",
            error=error,
            duration_s=time.monotonic() - started,
        )

    def _execute(self, call: ToolCall, ctx: ToolContext) -> dict:
        deadline = min(ctx.deadline_monotonic, time.monotonic() + ctx.limits.timeout_s)
        total = 0
        allowed = {a.path: a for a in ctx.source_artifacts}

        def check() -> None:
            if time.monotonic() >= deadline:
                raise OperationFailed(ErrorInfo(code="timeout", message="local tool deadline"))

        def read(raw: str) -> str:
            nonlocal total
            check()
            path = parse_rel_path(raw)
            if path not in allowed or path.suffix.casefold() not in (".md", ".csv", ".txt"):
                raise OperationFailed(
                    ErrorInfo(code="denied", message="artifact unavailable to local reader")
                )
            if total + allowed[path].size_bytes > min(ctx.limits.max_read_bytes, 8 * 1024 * 1024):
                raise OperationFailed(ErrorInfo(code="resource_limit", message="tool read limit"))
            content = bytearray()
            with (
                open_directory(ctx.workspace_root) as root,
                artifact_file(root, path) as descriptor,
            ):
                if os.fstat(descriptor).st_size != allowed[path].size_bytes:
                    raise OperationFailed(
                        ErrorInfo(code="denied", message="source artifact changed")
                    )
                while chunk := os.read(descriptor, 65_536):
                    check()
                    total += len(chunk)
                    if total > min(ctx.limits.max_read_bytes, 8 * 1024 * 1024):
                        raise OperationFailed(
                            ErrorInfo(code="resource_limit", message="tool read limit")
                        )
                    content.extend(chunk)
            if (
                len(content) != allowed[path].size_bytes
                or hashlib.sha256(content).hexdigest() != allowed[path].sha256
            ):
                raise OperationFailed(ErrorInfo(code="denied", message="source artifact changed"))
            text = content.decode("utf-8-sig")
            if "\x00" in text:
                raise ValueError("NUL in text artifact")
            return text

        check()
        args = call.arguments
        evidence: list[str] = []
        if call.tool_name == "local_math":
            answer = arithmetic(args, check)
        elif call.tool_name in ("query_csv", "audit_csv", "read_file"):
            path = args["path"]
            if not isinstance(path, str):
                raise ValueError("invalid path")
            text = read(path)
            if call.tool_name == "read_file":
                if set(args) != {"path"}:
                    raise ValueError("invalid read arguments")
                check()
                return {"text": text, "path": path}
            if PurePosixPath(path).suffix.casefold() != ".csv":
                raise ValueError("CSV required")
            if call.tool_name == "audit_csv":
                from nsu_budget_agent.validation.csv_audit import audit_csv

                try:
                    valid, independent_answer = audit_csv(text, args, deadline)
                except TimeoutError as exc:
                    raise OperationFailed(
                        ErrorInfo(code="timeout", message="CSV audit deadline")
                    ) from exc
                return {"valid": valid, "answer": independent_answer}
            answer, evidence = csv_query(text, args, check), [path]
        elif call.tool_name == "lookup_markdown":
            raw_paths = args["paths"]
            if not isinstance(raw_paths, list):
                raise ValueError("invalid profile list")
            paths: list[str] = []
            for raw_path in raw_paths:
                if not isinstance(raw_path, str):
                    raise ValueError("invalid profile path")
                paths.append(raw_path)
            if (
                not isinstance(paths, list)
                or not 0 < len(paths) <= 256
                or any(not isinstance(p, str) for p in paths)
                or len(set(paths)) != len(paths)
            ):
                raise ValueError("invalid profile list")
            if any(PurePosixPath(p).suffix.casefold() != ".md" for p in paths):
                raise ValueError("Markdown required")
            answer, evidence = markdown_lookup({p: read(p) for p in paths}, args, check)
        else:
            raise OperationFailed(
                ErrorInfo(code="unsupported", message="local tool is unavailable")
            )
        check()
        return {"answer": answer, "evidence_paths": evidence}
