"""Process-wide coordination of trusted agent operations on original task trees.

Original-tree writers must hold source_operation through validation and commit.
This is not a filesystem lock against other processes or uncooperative code.
"""

from contextlib import contextmanager
from threading import RLock
from typing import Iterator

_source_lock = RLock()


@contextmanager
def source_operation() -> Iterator[None]:
    with _source_lock:
        yield
