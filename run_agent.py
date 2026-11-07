"""A5-lite entry point. Resolve project code independently of --tasks-dir."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nsu_budget_agent.orchestration.cli import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
