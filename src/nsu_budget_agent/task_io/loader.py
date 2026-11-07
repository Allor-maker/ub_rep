"""Per-task failures and duplicate IDs are isolated in LoadReport."""

import asyncio
import os
import stat
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

from nsu_budget_agent.contracts.models import (
    ContractError,
    ErrorInfo,
    LoadIssue,
    LoadReport,
    OperationFailed,
    TaskSpec,
)
from nsu_budget_agent.contracts.ports import TaskAdapter

from ._files import excluded, open_directory
from .adapters import OrganizerTaskAdapter


class DirectoryTaskLoader:
    def __init__(self, adapter: TaskAdapter | None = None) -> None:
        self._adapter = adapter if adapter is not None else OrganizerTaskAdapter()

    @staticmethod
    def _validate_roots(path: Path, task: TaskSpec, expected: os.stat_result) -> None:
        try:
            work = task.work_root
            if task.task_root != path or ".." in work.parts:
                raise ContractError("invalid adapter roots")
            relative = work.relative_to(path)
            with open_directory(path, expected=expected) as root, ExitStack() as stack:
                current = root
                for name in relative.parts:
                    if excluded(name):
                        raise ContractError("excluded work root")
                    current = stack.enter_context(current.child(name))
        except (ValueError, OSError):
            raise OperationFailed(
                ErrorInfo(code="invalid_input", message="unsafe adapter task or work root")
            ) from None

    @staticmethod
    def _discover(
        tasks_dir: Path,
    ) -> tuple[tuple[tuple[Path, os.stat_result], ...], list[LoadIssue]]:
        paths: list[tuple[Path, os.stat_result]] = []
        issues: list[LoadIssue] = []
        try:
            with open_directory(tasks_dir) as root:
                for name in root.names():
                    if excluded(name):
                        continue
                    try:
                        info = root.info(name)
                        if stat.S_ISDIR(info.st_mode):
                            paths.append((root.path / name, info))
                    except (OSError, ContractError):
                        issues.append(
                            LoadIssue(
                                task_root=root.path / name,
                                error=ErrorInfo(
                                    code="invalid_input", message="unsafe task directory entry"
                                ),
                            )
                        )
        except (OSError, ContractError):
            raise OperationFailed(
                ErrorInfo(code="invalid_input", message="invalid tasks directory")
            ) from None
        return tuple(paths), issues

    async def load(self, tasks_dir: Path) -> LoadReport:
        paths, issues = await asyncio.to_thread(self._discover, tasks_dir)
        tasks: list[TaskSpec] = []
        for path, expected in paths:
            try:
                if type(self._adapter) is OrganizerTaskAdapter:
                    task = await self._adapter._load_checked(path, expected)
                else:
                    task = await self._adapter.load_task(path)
                await asyncio.to_thread(self._validate_roots, path, task, expected)
                tasks.append(task)
            except OperationFailed as exc:
                issues.append(LoadIssue(task_root=path, error=exc.error))
        counts = Counter(task.task_id for task in tasks)
        accepted: list[TaskSpec] = []
        for task in tasks:
            if counts[task.task_id] != 1:
                issues.append(
                    LoadIssue(
                        task_root=task.task_root,
                        error=ErrorInfo(code="invalid_input", message="duplicate task ID"),
                    )
                )
            else:
                accepted.append(task)
        return LoadReport(tasks=tuple(accepted), issues=tuple(issues))
