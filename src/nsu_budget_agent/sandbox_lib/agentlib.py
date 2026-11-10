"""Tiny helpers importable by model-written scripts (``from agentlib import kb, rows``).

They exist to cut OUTPUT tokens: a 3-line script using them replaces a 30-line parser.
Pure stdlib, no access to anything outside the script's working directory.
"""

import csv
import os
import re


def _norm(key):
    return re.sub(r"\s+", " ", str(key)).strip().casefold()


class CIDict(dict):
    """dict that keeps the ORIGINAL keys (iteration, printing, comparisons) but looks
    them up case- and whitespace-insensitively: d["sergei garcia"] finds "Sergei Garcia"."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._index = {}
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def _key(self, key):
        return self._index.get(_norm(key), key)

    def __missing__(self, key):
        known = ", ".join(repr(k) for k in list(self)[:30])
        raise KeyError(f"{key!r} not found; available keys: {known}")

    def __setitem__(self, key, value):
        old = self._index.get(_norm(key))
        if old is not None and old != key:
            super().__delitem__(old)
        self._index[_norm(key)] = key
        super().__setitem__(key, value)

    def __getitem__(self, key):
        return super().__getitem__(self._key(key))

    def __contains__(self, key):
        return _norm(key) in self._index

    def __delitem__(self, key):
        real = self._key(key)
        super().__delitem__(real)
        self._index.pop(_norm(real), None)

    def get(self, key, default=None):
        return super().get(self._key(key), default)


_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$", re.M)
_FIELD = re.compile(r"^\s*[-*]?\s*\**([^:#|\n*]{1,80}?)\**\s*:\s*(.+?)\s*$", re.M)
_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_SEP = re.compile(r"^\s*\|?\s*:?-{2,}")


def _cells(line):
    """Cells of a table line with or without outer pipes ('| a | b |' or 'a | b')."""
    body = line.strip()
    if body.startswith("|"):
        body = body[1:]
    if body.endswith("|"):
        body = body[:-1]
    return [c.strip() for c in body.split("|")]


def _is_table_line(line):
    """'| a | b |' or a bare 'a | b | c' line (at least two pipes, i.e. three cells)."""
    return bool(_ROW.match(line)) or (line.count("|") >= 2 and not line.lstrip().startswith(("#", ">", "```")))


def _md_tables(text):
    """Markdown tables in text -> list of (header or None, rows as lists of cells).

    Also reads 'bare' pipe tables (no outer pipes, no --- separator line): their first line is
    the header when none of its cells is a number."""
    out, block = [], []
    for line in text.splitlines() + [""]:
        if _is_table_line(line):
            block.append(line)
            continue
        if block:
            rows = [_cells(b) for b in block if not _SEP.match(b)]
            header = None
            if len(block) > 1 and _SEP.match(block[1]):
                header, rows = rows[0], rows[1:]
            elif len(rows) > 1 and not _ROW.match(block[0]) and all(
                    c and _num(c) == c for c in rows[0]):
                header, rows = rows[0], rows[1:]
            if rows:
                out.append((header, rows))
            block = []
    return out


class Row(CIDict):
    """A table row: row["Person"] (case-insensitive header) or row[0] (column position)."""

    def __init__(self, cells, header, **extra):
        super().__init__({**dict(zip(header, cells, strict=False)), **extra})
        self.cells = list(cells)

    def __getitem__(self, key):
        if isinstance(key, int) and not super().__contains__(key):
            return self.cells[key]
        return super().__getitem__(key)

    def get(self, key, default=None):
        if isinstance(key, int) and not super().__contains__(key):
            return self.cells[key] if -len(self.cells) <= key < len(self.cells) else default
        return super().get(key, default)


def tables(path="."):
    """Rows of every markdown table under path (a file or a directory), as one flat list.

    row["Header Name"] or row[0] (position); numbers converted; row["file"] = source
    file name, row["title"] = the file's first heading. Header-less tables: keys 0, 1, ...
    """
    files = [path] if os.path.isfile(path) else [
        os.path.join(r, n) for r, _d, ns in os.walk(path) for n in sorted(ns) if n.lower().endswith((".md", ".txt"))]
    result = []
    for name in files:
        with open(name, encoding="utf-8", errors="replace") as fh:
            text = fh.read()
        heading = _HEADING.search(text)
        title = heading.group(1).strip() if heading else ""
        for header, rows in _md_tables(text):
            keys = header or list(range(max((len(r) for r in rows), default=0)))
            for r in rows:
                result.append(Row([_num(c) for c in r], keys, file=os.path.basename(name), title=title))
    return result


def kb(path="kb"):
    """Markdown knowledge base -> CIDict: entity name -> CIDict of fields.

    Each file: '# Entity Name' heading + 'Key: Value' lines or a 2-column
    'Field | Value' table. Extra keys: 'name', 'file'. For other layouts
    (multi-column tables, prose) use tables() or parse the text directly.
    Duplicate names keep the first file and are listed in kb(...).duplicates.
    """
    result = CIDict()
    result.duplicates = []
    for root, _dirs, files in os.walk(path):
        for name in sorted(files):
            if not name.lower().endswith((".md", ".markdown", ".txt")):
                continue
            with open(os.path.join(root, name), encoding="utf-8", errors="replace") as fh:
                text = fh.read()
            heading = _HEADING.search(text)
            if not heading:
                continue
            entity = heading.group(1).strip()
            fields = CIDict(name=entity, file=name)
            for key, value in _FIELD.findall(text):
                if _norm(key) not in fields:
                    fields[key] = value.strip()
            for header, rows in _md_tables(text):  # 2-column "Field | Value" tables
                if all(len(r) == 2 for r in rows) and (header is None or len(header) == 2):
                    for key, value in rows:
                        if key and _norm(key) not in fields:
                            fields[key] = value
            if entity in result:
                result.duplicates.append(entity)
                continue
            result[entity] = fields
    return result


def _num(value):
    text = value.strip()
    if re.fullmatch(r"[-+]?\d+", text):
        return int(text)
    if re.fullmatch(r"[-+]?(\d+\.\d*|\.\d+)([eE][-+]?\d+)?", text):
        return float(text)
    return value


def rows(path, delimiter=None):
    """CSV/TSV -> list of CIDict rows; integer/float cells converted to numbers."""
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as fh:
        sample = fh.read(4096)
        fh.seek(0)
        if delimiter is None:
            delimiter = "\t" if sample.count("\t") > sample.count(",") else ","
        return [CIDict({k: _num(v or "") for k, v in r.items() if k is not None})
                for r in csv.DictReader(fh, delimiter=delimiter)]
