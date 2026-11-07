"""Read-only, anchored traversal. No bare path.open after a containment check."""

import os
import stat
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path, PureWindowsPath
from typing import Iterator

from nsu_budget_agent.contracts.models import (
    ContractError,
    ErrorInfo,
    OperationFailed,
    parse_rel_path,
)


def safe_name(name: str) -> None:
    parse_rel_path(name)
    if ":" in name or "/" in name or PureWindowsPath(name).is_reserved():
        raise ContractError("unsafe file name")
    if name.endswith((".", " ")) or any(ord(char) < 32 for char in name):
        raise ContractError("unsafe file name")


def excluded(name: str) -> bool:
    lower = name.casefold()
    return (
        lower.startswith(".")
        or lower in ("task.json", "answer.txt", "__pycache__", "__macosx", "node_modules")
        or any(
            lower == label
            or (
                lower.startswith(label)
                and (not lower[len(label)].isalnum() or lower[len(label)].isdecimal())
            )
            for label in ("expected", "gold")
        )
    )


def check_stat(info: os.stat_result) -> None:
    if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
        raise ContractError("symlinks and reparse points are forbidden")
    if not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode)):
        raise ContractError("only regular files and directories are allowed")
    if info.st_ino == 0:
        raise ContractError("file identity is unavailable")
    if stat.S_ISREG(info.st_mode) and info.st_nlink != 1:
        raise ContractError("hard links are forbidden")


class Directory:
    def __init__(self, path: Path, descriptor: int) -> None:
        self.path = path
        self.descriptor = descriptor

    def names(self, *, limit: int = 10_000) -> list[str]:
        names: list[str] = []
        with os.scandir(self.path if sys.platform == "win32" else self.descriptor) as entries:
            for entry in entries:
                if len(names) >= limit:
                    raise OperationFailed(
                        ErrorInfo(code="resource_limit", message="directory entry limit exceeded")
                    )
                names.append(entry.name)
        return sorted(names)

    def info(self, name: str) -> os.stat_result:
        safe_name(name)
        if sys.platform == "win32":
            result = (self.path / name).lstat()
        else:
            result = os.stat(name, dir_fd=self.descriptor, follow_symlinks=False)
        check_stat(result)
        return result

    @contextmanager
    def child(self, name: str, *, expected: os.stat_result | None = None) -> Iterator["Directory"]:
        safe_name(name)
        before = self.info(name)
        if not stat.S_ISDIR(before.st_mode):
            raise ContractError("directory required")
        if expected is not None and not os.path.samestat(expected, before):
            raise ContractError("directory changed before opening")
        if sys.platform == "win32":
            from ._windows import close_handle, handle_identity, open_handle

            descriptor = open_handle(self.path / name, directory=True)
            try:
                if handle_identity(descriptor) != (before.st_dev, before.st_ino):
                    raise ContractError("directory changed while opening")
                if not os.path.samestat(before, self.info(name)):
                    raise ContractError("directory path changed while opening")
                yield Directory(self.path / name, descriptor)
            finally:
                close_handle(descriptor)
        else:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=self.descriptor
            )
            try:
                if not os.path.samestat(before, os.fstat(descriptor)):
                    raise ContractError("directory changed while opening")
                if not os.path.samestat(before, self.info(name)):
                    raise ContractError("directory path changed while opening")
                yield Directory(self.path / name, descriptor)
            finally:
                os.close(descriptor)

    @contextmanager
    def file(self, name: str, *, expected: os.stat_result | None = None) -> Iterator[int]:
        before = self.info(name)
        if not stat.S_ISREG(before.st_mode):
            raise ContractError("regular file required")
        if expected is not None and not os.path.samestat(expected, before):
            raise ContractError("file changed before opening")
        if sys.platform == "win32":
            import msvcrt

            from ._windows import close_handle, open_handle

            handle = open_handle(self.path / name, directory=False)
            try:
                descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
            except BaseException:
                close_handle(handle)
                raise
        else:
            descriptor = os.open(
                name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self.descriptor
            )
        try:
            opened = os.fstat(descriptor)
            check_stat(opened)
            if not stat.S_ISREG(opened.st_mode) or not os.path.samestat(before, opened):
                raise ContractError("file changed before opening")
            yield descriptor
            after = os.fstat(descriptor)
            if (opened.st_size, opened.st_mtime_ns, opened.st_ctime_ns) != (
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            ):
                raise ContractError("file changed while reading")
        finally:
            os.close(descriptor)


@contextmanager
def open_directory(path: Path, *, expected: os.stat_result | None = None) -> Iterator[Directory]:
    if ".." in path.parts:
        raise ContractError("parent traversal is forbidden")
    absolute = path.absolute()
    with ExitStack() as stack:
        if sys.platform == "win32":
            from ._windows import close_handle, open_handle

            descriptor = open_handle(Path(absolute.anchor), directory=True)
            stack.callback(close_handle, descriptor)
        elif os.name == "posix":
            descriptor = os.open(absolute.anchor, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            stack.callback(os.close, descriptor)
        else:
            raise ContractError("safe filesystem backend is unavailable")
        current = Directory(Path(absolute.anchor), descriptor)
        parts = absolute.parts[1:]
        for index, name in enumerate(parts):
            current = stack.enter_context(
                current.child(name, expected=expected if index == len(parts) - 1 else None)
            )
        if not parts and expected is not None:
            if sys.platform == "win32":
                from ._windows import handle_identity

                if handle_identity(descriptor) != (expected.st_dev, expected.st_ino):
                    raise ContractError("root directory changed")
            elif not os.path.samestat(expected, os.fstat(descriptor)):
                raise ContractError("root directory changed")
        yield current
