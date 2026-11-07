"""Opt-in A1 SDK compatibility probe; not the A5 AttemptEngine implementation.

Run as a separate process: python -m nsu_budget_agent.orchestration.sdk_probe.
No models, provider configuration, task data, or external network are used.
"""

import asyncio
import importlib.metadata
import json
import os
import platform
import sys


def _offline_audit(event, args):
    # Windows asyncio may use loopback sockets internally. External connections
    # and name resolution are forbidden throughout SDK import and execution.
    if event == "socket.connect":
        address = args[1]
        if isinstance(address, tuple) and address[0] not in ("127.0.0.1", "::1"):
            raise RuntimeError("SDK probe forbids external connections")
    elif event == "socket.getaddrinfo":
        if args[0] not in (None, "localhost", "127.0.0.1", "::1"):
            raise RuntimeError("SDK probe forbids external name resolution")


async def probe() -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("A1 probe requires Python 3.11")
    if importlib.metadata.version("openjiuwen") != "0.1.19":
        raise RuntimeError("A1 probe requires exactly openjiuwen 0.1.19")

    # Keep every SDK import in orchestration, including compatibility checks.
    from openjiuwen.core.runner import Runner
    from openjiuwen.core.workflow import (
        End,
        Start,
        Workflow,
        WorkflowCard,
        WorkflowComponent,
        WorkflowExecutionState,
        create_workflow_session,
    )

    calls = []

    class Increment(WorkflowComponent):
        async def invoke(self, inputs, session, context):
            calls.append(inputs["value"])
            return {"value": inputs["value"] + 1}

    class Block(WorkflowComponent):
        def __init__(self, started, cleaned):
            super().__init__()
            self.started = started
            self.cleaned = cleaned

        async def invoke(self, inputs, session, context):
            self.started.set()
            try:
                await asyncio.Event().wait()
            finally:
                self.cleaned.set()

    def workflow(name, components):
        flow = Workflow(card=WorkflowCard(id=name, name=name, version="1"))
        flow.set_start_comp("start", Start(), inputs_schema={"value": "${value}"})
        previous = "start"
        for index, component in enumerate(components):
            current = f"step{index}"
            flow.add_workflow_comp(
                current,
                component,
                max_retries=0,
                inputs_schema={"value": "${" + previous + ".value}"},
            )
            flow.add_connection(previous, current)
            previous = current
        flow.set_end_comp("end", End(), inputs_schema={"value": "${" + previous + ".value}"})
        flow.add_connection(previous, "end")
        return flow

    try:
        flow = workflow("a1-success", [Increment(), Increment()])
        async with asyncio.timeout(5):
            result = await flow.invoke({"value": 1}, create_workflow_session())
        if result.state != WorkflowExecutionState.COMPLETED:
            raise AssertionError(f"unexpected state: {result.state}")
        if result.result != {"output": {"value": 3}} or calls != [1, 2]:
            raise AssertionError(f"unexpected result/calls: {result.result}, {calls}")

        for mode in ("cancel", "deadline"):
            started, cleaned = asyncio.Event(), asyncio.Event()
            flow = workflow(f"a1-{mode}", [Block(started, cleaned)])
            task = asyncio.create_task(flow.invoke({"value": 1}, create_workflow_session()))
            try:
                await asyncio.wait_for(started.wait(), timeout=5)
                if mode == "cancel":
                    task.cancel()
                    try:
                        await asyncio.wait_for(task, timeout=3)
                    except asyncio.CancelledError:
                        pass
                    else:
                        raise AssertionError("SDK swallowed cancellation")
                else:
                    try:
                        async with asyncio.timeout_at(asyncio.get_running_loop().time() + 0.05):
                            await task
                    except TimeoutError:
                        pass
                    else:
                        raise AssertionError("SDK swallowed deadline")
                await asyncio.wait_for(cleaned.wait(), timeout=3)
            finally:
                if not task.done():
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
        return {
            "python": platform.python_version(),
            "platform": platform.system(),
            "openjiuwen": "0.1.19",
            "two_components": "passed",
            "cancel_cleanup": "passed",
            "deadline_cleanup": "passed",
            "external_network": "blocked",
        }
    finally:
        await Runner.stop()


if __name__ == "__main__":
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    sys.addaudithook(_offline_audit)
    print(json.dumps(asyncio.run(probe()), ensure_ascii=False))
