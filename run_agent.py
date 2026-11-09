"""Official entry point: python /app/run_agent.py --tasks-dir /data/tasks"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from nsu_budget_agent.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
