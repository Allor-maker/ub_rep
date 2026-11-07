# Fixed Python 3.11 patch and OS family; see A6 report for digest/lock limitations.
FROM python:3.11.11-slim-bookworm AS dependencies

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN python -m venv "$VIRTUAL_ENV"
COPY requirements-runtime.txt /tmp/requirements-runtime.txt
RUN python -m pip install --no-compile --timeout 30 --retries 2 -r /tmp/requirements-runtime.txt \
    && python -m pip check \
    && find /opt/venv/lib/python3.11/site-packages -type d \( -name tests -o -name test -o -name __pycache__ \) -prune -exec rm -rf '{}' + \
    && find /opt/venv/lib/python3.11/site-packages -type f \( -name 'test_*.py' -o -name '*_test.py' -o -name '*.pyc' \) -delete

FROM dependencies AS app_payload

COPY src/nsu_budget_agent /project/nsu_budget_agent
COPY run_agent.py /payload/run_agent.py
# The final payload contains only runtime Python files, even if the source tree
# acquires unrelated data or test fixtures later.
RUN mkdir -p /payload/src \
    && cd /project \
    && find nsu_budget_agent -type f -name '*.py' ! -name 'test_*.py' ! -name '*_test.py' ! -name 'sdk_probe.py' -exec cp --parents -t /payload/src '{}' + \
    && test -f /payload/src/nsu_budget_agent/solvers/composition.py

FROM python:3.11.11-slim-bookworm AS runtime

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}" \
    PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HOME=/tmp/huggingface \
    HF_HUB_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

COPY --from=dependencies /opt/venv /opt/venv
COPY --from=app_payload /payload /app
RUN mkdir -p /data/tasks

# SDK 0.1.19 may write relative logs; /tmp is writable on the read-only rootfs.
WORKDIR /tmp
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
