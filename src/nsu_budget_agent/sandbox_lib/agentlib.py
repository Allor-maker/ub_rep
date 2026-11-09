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
    """dict with case- and whitespace-insensitive string keys."""

    def __init__(self, *args, **kwargs):
        super().__init__()
        for k, v in dict(*args, **kwargs).items():
            self[k] = v

    def __setitem__(self, key, value):
        super().__setitem__(_norm(key), value)

    def __getitem__(self, key):
        return super().__getitem__(_norm(key))

    def __contains__(self, key):
        return super().__contains__(_norm(key))

    def get(self, key, default=None):
        return super().get(_norm(key), default)


_HEADING = re.compile(r"^\s*#\s+(.+?)\s*$", re.M)
_FIELD = re.compile(r"^\s*[-*]?\s*([^:#\n]{1,80}?)\s*:\s*(.+?)\s*$", re.M)


def kb(path="kb"):
    """Markdown knowledge base -> CIDict: entity name -> CIDict of fields.

    Each file: '# Entity Name' heading + 'Key: Value' lines. Extra keys: 'name', 'file'.
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
