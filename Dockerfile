# Astra v2 harness on openJiuwen 0.1.19.
# Pinned base image tag and a full pip lock (requirements.lock, installed with --no-deps).
# Works with the official read-only launch: --read-only --tmpfs /tmp --workdir /app.
FROM python:3.11.11-slim-bookworm

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:${PATH}"

RUN python -m venv "$VIRTUAL_ENV"
COPY requirements.lock /tmp/requirements.lock
RUN pip install --no-compile --timeout 60 --retries 5 --no-deps -r /tmp/requirements.lock \
    && pip check \
    && find /opt/venv -type d -name __pycache__ -prune -exec rm -rf '{}' + \
    && rm /tmp/requirements.lock

COPY run_agent.py /app/run_agent.py
COPY src/nsu_budget_agent /app/src/nsu_budget_agent

ENV PYTHONPATH=/app/src \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    TMPDIR=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    MPLCONFIGDIR=/tmp/matplotlib

RUN mkdir -p /data/tasks && python -c "import nsu_budget_agent.runner"
WORKDIR /app
CMD ["python", "/app/run_agent.py", "--tasks-dir", "/data/tasks"]
