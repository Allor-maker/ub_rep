"""CLI runtime deadline defaults and explicit overrides."""

import os
import unittest
from unittest.mock import AsyncMock, patch

from nsu_budget_agent.orchestration.cli import main


class CliDeadlineTests(unittest.TestCase):
    def test_runtime_deadline_default_and_override(self):
        for arguments, expected in (([], 3300), (["--timeout-s", "2.5"], 2.5)):
            with self.subTest(arguments=arguments):
                run = AsyncMock(return_value=0)
                with patch.dict(os.environ), patch("nsu_budget_agent.orchestration.cli._run", run):
                    self.assertEqual(
                        main(["--tasks-dir", "tasks", *arguments], factory=lambda *_: None), 0
                    )
                self.assertEqual(run.await_args.args[0].timeout_s, expected)
