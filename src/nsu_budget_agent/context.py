"""Compact, token-cheap description of a task's files for the model.

Input tokens cost 1/10 of output tokens, so a good preview (header, sample rows,
distinct values, field keys of a knowledge base) is cheap and saves expensive
output tokens on guessing and repairs.
"""

from __future__ import annotations

import csv
import io
import re
from collections import defaultdict
from pathlib import Path

TEXT_SUFFIXES = {
    ".txt", ".md", ".csv", ".tsv", ".json", ".jsonl", ".py", ".yaml", ".yml", ".ini",
    ".cfg", ".toml", ".xml", ".html", ".log", ".sql", ".js", ".sh",
}
_SKIP = {"task.json", "answer.txt", "__pycache__", ".agent_solve.py"}
_FIELD_RE = re.compile(r"^\s*[-*]?\s*([A-Za-z0-9][\w ()/'-]{0,50}?)\s*:\s*\S", re.M)


def list_files(root: Path, limit: int = 2000) -> list[Path]:
    files: list[Path] = []
    if not root.is_dir():
        return files
    stack = [root]
    while stack and len(files) < limit:
        current = stack.pop()
        try:
            entries = sorted(current.iterdir())
        except OSError:
            continue
        for entry in entries:
            if entry.name in _SKIP or entry.is_symlink():
                continue
            if entry.is_dir():
                stack.append(entry)
            elif entry.is_file():
                files.append(entry)
    return sorted(files)


def read_text(path: Path, limit: int = 2_000_000) -> str:
    try:
        with path.open("rb") as fh:
            return fh.read(limit).decode("utf-8", "replace")
    except OSError:
        return ""


def _csv_preview(rel: str, text: str, delimiter: str = ",") -> str:
    rows = list(csv.reader(io.StringIO(text), delimiter=delimiter))
    if not rows:
        return f"{rel}: (empty)"
    header, body = rows[0], rows[1:]
    parts = [f"{rel}: {len(body)} data rows; header: {delimiter.join(header)}"]
    parts += ["  " + delimiter.join(r) for r in body[:3]]
    cats = []
    for i, name in enumerate(header):
        values = {r[i] for r in body if i < len(r)}
        numeric = all(re.fullmatch(r"-?\d+(\.\d+)?", v or "0") for v in values)
        if values and not numeric and len(values) <= 15:
            cats.append(f"{name}={{{', '.join(sorted(values))}}}")
        elif values and numeric and len(values) <= 15 and len(body) > 15:
            cats.append(f"{name}={{{', '.join(sorted(values, key=lambda v: float(v or 0)))}}}")
    if cats:
        parts.append("  distinct values: " + "; ".join(cats))
    return "\n".join(parts)


def _md_group(rel_dir: str, paths: list[Path], root: Path) -> str:
    names = [p.name for p in paths]
    shown = ", ".join(names[:60]) + (f", ... (+{len(names) - 60})" if len(names) > 60 else "")
    lines = [f"{rel_dir}/ ({len(paths)} files): {shown}"]
    keys: dict[str, int] = defaultdict(int)
    by_kind: dict[str, list[Path]] = defaultdict(list)
    texts = {}
    for p in paths:
        text = read_text(p, 20_000)
        texts[p] = text
        for key in set(_FIELD_RE.findall(text)):
            keys[key.strip()] += 1
        by_kind[p.name.split("_")[0] if "_" in p.name else p.suffix].append(p)
    if keys:
        lines.append("field keys used (with file counts): " + ", ".join(f"{k} ({n})" for k, n in sorted(keys.items())))
    samples = 0
    for group in by_kind.values():
        if samples >= 4:
            break
        richest = max(group, key=lambda p: len(set(_FIELD_RE.findall(texts[p]))))
        lines.append(f"--- example {richest.relative_to(root).as_posix()}:\n{texts[richest][:700].strip()}")
        samples += 1
    return "\n".join(lines)


def build_preview(root: Path, *, max_chars: int = 9000) -> str:
    files = list_files(root)
    if not files:
        return "(no files)"
    groups: dict[Path, list[Path]] = defaultdict(list)
    for f in files:
        groups[f.parent].append(f)
    parts: list[str] = []
    for directory, paths in sorted(groups.items()):
        rel_dir = directory.relative_to(root).as_posix()
        if len(paths) > 6 and all(p.suffix.lower() in (".md", ".txt") for p in paths):
            parts.append(_md_group(rel_dir if rel_dir != "." else ".", paths, root))
            continue
        for p in paths:
            rel = p.relative_to(root).as_posix()
            size = p.stat().st_size if p.exists() else 0
            suffix = p.suffix.lower()
            if suffix not in TEXT_SUFFIXES and size > 0:
                head = read_text(p, 200)
                if "\x00" in head:
                    parts.append(f"{rel}: binary file, {size} bytes")
                    continue
            text = read_text(p)
            if suffix in (".csv", ".tsv"):
                parts.append(_csv_preview(rel, text, "\t" if suffix == ".tsv" else ","))
            elif len(text) <= 2500:
                parts.append(f"{rel}:\n{text.strip()}")
            else:
                lines = text.splitlines()
                parts.append(f"{rel} ({len(lines)} lines, {size} bytes), first lines:\n" + "\n".join(lines[:20]))
    preview = "\n".join(parts)
    if len(preview) > max_chars:
        preview = preview[:max_chars] + "\n... (preview truncated; read the files for details)"
    return preview


# ---------------------------------------------------------------- knowledge base

_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$", re.M)
_KV = re.compile(r"^\s*[-*]?\s*([^:#\n]{1,60}?)\s*:\s*(.+?)\s*$", re.M)


def norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def parse_kb(paths: list[Path]) -> dict[str, list[dict[str, str]]]:
    """Map normalized entity name -> list of field dicts (a list to expose duplicates)."""
    kb: dict[str, list[dict[str, str]]] = defaultdict(list)
    for p in paths:
        text = read_text(p, 200_000)
        heading = _HEADING.search(text)
        if not heading:
            continue
        fields = {"__name__": heading.group(1).strip(), "__file__": p.name}
        for key, value in _KV.findall(text):
            fields.setdefault(norm(key), value.strip())
        kb[norm(heading.group(1))].append(fields)
    return dict(kb)
