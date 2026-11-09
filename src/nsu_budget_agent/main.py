"""Entry point. Never fails because of configuration: worst case it runs without the LLM."""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import sys
import tempfile
import traceback
import warnings
from dataclasses import replace


def _prepare_process() -> None:
    """openJiuwen writes ./logs at import; the official run has a read-only /app cwd."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    for candidate in (os.environ.get("TMPDIR"), "/tmp", tempfile.gettempdir()):
        if not candidate:
            continue
        try:
            workdir = tempfile.mkdtemp(prefix="astra-run-", dir=candidate)
            os.chdir(workdir)
            return
        except OSError:
            continue


def main(argv: list[str] | None = None) -> int:
    warnings.filterwarnings("ignore")  # third-party deprecation noise in the logs
    from .config import parse_config

    config = parse_config(argv)
    config = replace(
        config,
        tasks_dir=config.tasks_dir.absolute(),
        out=config.out.absolute() if config.out else None,
        log_dir=config.log_dir.absolute() if config.log_dir else None,
    )
    _prepare_process()
    real_stdout = sys.stdout
    try:
        # Framework loggers bind to sys.stdout lazily; keep stdout for the final summary only.
        with contextlib.redirect_stdout(sys.stderr):
            from .runner import Runner

            summary = asyncio.run(Runner(config).run())
    except KeyboardInterrupt:
        return 130
    except BaseException:  # noqa: BLE001 - answers already written stay valid
        traceback.print_exc()
        return 1
    print(json.dumps(summary, ensure_ascii=False), file=real_stdout, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
