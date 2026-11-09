"""Run orchestration: triage -> plan -> first pass -> second pass -> report.

1. Triage (0 tokens): exact-template fast path answers or rejects (infeasible) tasks.
2. Plan: estimate tokens per remaining task; order by reward / estimate.
3. First pass: main model, one bounded attempt per task, answer.txt written at once.
4. Second pass (time and budget permitting): tasks the first pass could not answer get a
   fresh attempt from a second model (a different model = an independent opinion).
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import planner, workflow
from .config import Config
from .llm import Ledger, LLMClient, LLMError, LLMUnavailable
from .publish import Publisher, atomic_write
from .solvers import coding, fastpath, precheck, program
from .solvers.common import SolveResult
from .tasks import Task, load_tasks
from .trace import Trace


@dataclass
class Record:
    id: str
    type: str
    reward: float
    status: str = "pending"  # answered | infeasible | failed | pending | skipped
    answer: str = ""
    source: str = ""
    reason: str = ""
    estimate: float = 0.0
    cost: float = 0.0
    calls: int = 0
    seconds: float = 0.0
    attempts: list[str] = field(default_factory=list)


class Runner:
    def __init__(self, config: Config, *, transport=None) -> None:
        self.config = config
        self.started = time.monotonic()
        self.deadline = self.started + config.deadline_s
        self.ledger = Ledger(budget=config.budget)
        self.llm: LLMClient | None = None
        if config.llm_enabled:
            self.llm = LLMClient(
                config.base_url or "",
                config.api_key or "",
                ledger=self.ledger,
                timeout_s=config.llm_timeout_s,
                max_parallel=config.concurrency,
                deadline=self.deadline - 20,
                transport=transport,
            )
        self.publisher = Publisher(config.out)
        self.records: dict[str, Record] = {}
        self.traces: dict[str, Trace] = {}
        self.engines: dict[str, int] = {}
        self._log_lock = asyncio.Lock()
        self.codes: dict[str, str] = {}  # task id -> script behind the published LLM answer
        self.reward_mid = 0.0  # median reward of the run
        self.reward_high = 0.0  # reward from which a task counts as high-value
        self.verify_from = float("inf")  # reward from which LLM answers are double-checked

    # ------------------------------------------------------------------ helpers
    def _log(self, message: str) -> None:
        elapsed = time.monotonic() - self.started
        print(f"[astra {elapsed:6.1f}s] {message}", file=sys.stderr, flush=True)

    def _time_left(self) -> float:
        return self.deadline - time.monotonic()

    def _finish(self, task: Task, result: SolveResult, source: str) -> None:
        record = self.records[task.id]
        record.attempts.append(f"{source}:{result.status}")
        if result.status == "answer" and result.answer:
            if result.files:
                self.publisher.apply_files(task, result.files)
            self.publisher.publish(task, result.answer)
            record.status, record.answer, record.source = "answered", result.answer, source
            if result.code:
                self.codes[task.id] = result.code
            record.reason = result.reason
        elif result.status == "infeasible":
            record.status, record.source, record.reason = "infeasible", source, result.reason
        else:
            record.status, record.source, record.reason = "failed", source, result.reason
        trace = self.traces[task.id]
        record.cost, record.calls = round(trace.cost, 1), trace.calls
        self._log(
            f"{task.id:<24} {record.status:<10} {source:<34} cost={record.cost:7.1f} "
            f"total={self.ledger.spent:9.1f} {('-> ' + record.answer[:40]) if record.answer else record.reason[:60]}"
        )

    async def _phase(self, name: str, task: Task, solve_step, finish=None) -> None:
        """Run one task phase as an openJiuwen workflow: solve -> publish."""
        finish = finish or self._finish
        holder: dict[str, tuple[SolveResult, str]] = {}

        async def solve() -> None:
            holder["result"] = await solve_step()

        async def publish() -> None:
            if "result" in holder:
                result, source = holder["result"]
                finish(task, result, source)

        started = time.monotonic()
        run = await workflow.run_phase(f"{name}:{task.id}", [("solve", solve), ("publish", publish)])
        self.engines[run.engine] = self.engines.get(run.engine, 0) + 1
        self.records[task.id].seconds += round(time.monotonic() - started, 2)
        error = run.errors.get("solve")
        if isinstance(error, LLMUnavailable):
            self.records[task.id].reason = "llm unavailable"
            return
        if error is not None:
            if not isinstance(error, (LLMError, asyncio.TimeoutError, TimeoutError)):
                self._log(f"{task.id}: {name} error {type(error).__name__}: {error}")
            record = self.records[task.id]
            if record.status == "pending":
                record.status, record.reason = "failed", f"{type(error).__name__}: {str(error)[:120]}"
        if "publish" in run.errors:
            self._log(f"{task.id}: publish error {run.errors['publish']!r}")

    # ------------------------------------------------------------------ phases
    async def _triage(self, task: Task) -> None:
        trace = self.traces[task.id]
        fast = fastpath.solve(task)
        if fast is not None:
            trace.note("fastpath", status=fast.status, reason=fast.reason)
            result = (
                SolveResult("answer", fast.answer, reason=fast.reason)
                if fast.status == "answer"
                else SolveResult("infeasible", reason=fast.reason)
            )

            async def fast_step():
                return result, "fastpath"

            await self._phase("triage", task, fast_step)
        elif (
            self.config.reward_aware
            and not task.is_coding
            and task.reward < self.reward_high
            and (missing := precheck.missing_names(task))
        ):
            reason = "not in knowledge base: " + ", ".join(missing)[:200]
            trace.note("precheck", status="infeasible", reason=reason)

            async def precheck_step():
                return SolveResult("infeasible", reason=reason), "precheck"

            await self._phase("triage", task, precheck_step)
        elif task.is_coding and self.llm is None:

            async def local_step():
                return await coding.solve(task, None, None, trace), "local-tests"

            await self._phase("triage", task, local_step)

    async def _attempt(self, task: Task, model: str, phase: str, repairs: int) -> None:
        if self.llm is None:
            return
        trace = self.traces[task.id]
        timeout = max(5.0, min(240.0, self._time_left() - 15))
        # Cap one task's spend relative to its estimate (protects against repair loops).
        cap = max(600.0, 6 * self.records[task.id].estimate)

        async def solve_step():
            if task.is_coding:
                coro = coding.solve(
                    task, self.llm, model, trace, repairs=self.config.coding_repairs, cost_cap=cap
                )
            else:
                coro = program.solve(
                    task, self.llm, model, trace, repairs=repairs,
                    timeout=self.config.sandbox_timeout_s, cost_cap=cap,
                )
            return await asyncio.wait_for(coro, timeout=timeout), f"{phase}:{model}"

        await self._phase(phase, task, solve_step)

    def _wants_second_opinion(self, task: Task) -> bool:
        record = self.records[task.id]
        if record.source in ("fastpath", "precheck") or record.status not in ("failed", "infeasible", "pending"):
            return False
        if record.reason in ("estimated cost exceeds remaining budget", "llm unavailable"):
            return False
        if self.config.reward_aware and record.status == "infeasible" and task.reward < self.reward_mid:
            return False  # low value: accept the model's UNSOLVABLE without paying for a second opinion
        if task.is_coding:
            # Coding is verified by tests; a model's UNSOLVABLE or a no-progress stop on
            # (likely contradictory) tests is not worth another expensive multi-call attempt.
            if record.status != "failed" or record.reason.startswith("per-task"):
                return False
            # "no progress" is typical for contradictory tests, but a hard real bug looks the same:
            # worth one stronger attempt only when the task is valuable enough.
            return not record.reason.startswith("no progress") or task.reward >= self.reward_mid
        return True

    @staticmethod
    def _same(a: str, b: str) -> bool:
        x, y = a.strip().casefold(), b.strip().casefold()
        if x == y:
            return True
        try:
            return abs(float(x) - float(y)) < 1e-9
        except ValueError:
            return False

    def _wants_verification(self, task: Task) -> bool:
        record = self.records[task.id]
        return (
            record.status == "answered"
            and record.source.startswith("pass1:")
            and not task.is_coding  # coding answers are already verified by the tests
            and (task.reward >= self.verify_from or task.type in self.config.verify_types)
        )

    async def _verify(self, task: Task, model: str) -> None:
        """Independent solve by the second model. On disagreement the same model JUDGES the two
        programs side by side (spotting a bug in a comparison is easier than avoiding it, and a
        re-solve repeats systematic mistakes); the answer changes only if the judge picks the new one."""
        if self.llm is None:
            return
        trace = self.traces[task.id]
        record = self.records[task.id]
        cap = max(600.0, 6 * record.estimate)

        async def solve_once(temperature: float) -> SolveResult:
            coro = program.solve(task, self.llm, model, trace, repairs=1, timeout=self.config.sandbox_timeout_s,
                                 cost_cap=cap, temperature=temperature)
            return await asyncio.wait_for(coro, timeout=max(5.0, min(240.0, self._time_left() - 15)))

        async def solve_step():
            first = await solve_once(0.0)
            if first.status != "answer" or not first.answer or self._same(first.answer, record.answer):
                return first, f"verify:{model}"
            original = self.codes.get(task.id)
            if not original or not first.code:
                return SolveResult("failed", reason="no programs to compare"), f"verify:{model}"
            verdict = await program.judge(task, self.llm, model, trace, (original, record.answer),
                                          (first.code, first.answer))
            trace.note("judge", original=record.answer, verifier=first.answer, verdict=verdict)
            if verdict == "B":
                return first, f"verify:{model}"
            return SolveResult("failed", reason=f"judge kept the original ({verdict})"), f"verify:{model}"

        def finish(task: Task, result: SolveResult, source: str) -> None:
            record.attempts.append(f"{source}:{result.status}")
            record.cost, record.calls = round(trace.cost, 1), trace.calls
            if result.status == "answer" and result.answer and not self._same(result.answer, record.answer):
                self._log(f"{task.id:<24} verify changed {record.answer[:30]!r} -> {result.answer[:30]!r}")
                self.publisher.publish(task, result.answer)
                record.answer, record.source = result.answer, source
            else:
                self._log(f"{task.id:<24} verify {'agrees' if result.status == 'answer' else 'kept'} "
                          f"cost={record.cost:7.1f} total={self.ledger.spent:9.1f}")

        await self._phase("verify", task, solve_step, finish)

    async def _pool(self, tasks: list[Task], fn) -> None:
        queue: asyncio.Queue[Task] = asyncio.Queue()
        for task in tasks:
            queue.put_nowait(task)

        async def worker() -> None:
            while not queue.empty():
                if self._time_left() < 45 or (self.llm is not None and not self.llm.available):
                    return
                task = queue.get_nowait()
                record = self.records[task.id]
                if not self.ledger.can_afford(record.estimate):
                    if record.status != "answered":  # never downgrade an answer we already have
                        record.status, record.reason = "skipped", "estimated cost exceeds remaining budget"
                    continue
                try:
                    await fn(task)
                except Exception as exc:  # noqa: BLE001 - one task never stops the run
                    self._log(f"{task.id}: unexpected {type(exc).__name__}: {exc}")

        await asyncio.gather(*(worker() for _ in range(self.config.concurrency)))

    # ------------------------------------------------------------------ main
    async def run(self) -> dict:
        tasks = load_tasks(self.config.tasks_dir)
        self._log(
            f"loaded {len(tasks)} tasks from {self.config.tasks_dir}; llm="
            f"{'on' if self.llm else 'off'} model={self.config.model} engine={workflow.sdk_status()}"
        )
        for task in tasks:
            self.records[task.id] = Record(id=task.id, type=task.type, reward=task.reward)
            self.traces[task.id] = Trace(task.id)
        rewards = sorted(t.reward for t in tasks) or [0.0]

        def quantile(q: float) -> float:
            return rewards[min(len(rewards) - 1, int(q * len(rewards)))]

        self.reward_mid, self.reward_high = quantile(0.5), quantile(0.75)
        if self.config.verify_top > 0:
            self.verify_from = quantile(1 - self.config.verify_top)
        try:
            for task in tasks:  # free, CPU-only
                try:
                    await self._triage(task)
                except Exception as exc:  # noqa: BLE001
                    self._log(f"{task.id}: triage error {type(exc).__name__}: {exc}")

            pending = [t for t in tasks if self.records[t.id].status == "pending"]
            for task in pending:
                try:
                    self.records[task.id].estimate = planner.estimate(task)
                except Exception:  # noqa: BLE001
                    self.records[task.id].estimate = 400.0
            estimates = {t.id: self.records[t.id].estimate for t in pending}
            first = planner.order(pending, estimates)
            self._log(
                f"triage: {sum(r.status == 'answered' for r in self.records.values())} answered free, "
                f"{sum(r.status == 'infeasible' for r in self.records.values())} infeasible, "
                f"{len(first)} for LLM (est. {sum(estimates.values()):.0f} tokens)"
            )
            if self.llm is not None:
                await self._pool(
                    first,
                    lambda t: self._attempt(t, self.config.model, "pass1", self.config.max_repairs),
                )
                second_model = self.config.second_model or self.config.model
                retry = [t for t in tasks if self._wants_second_opinion(t)]
                retry.sort(key=lambda t: -t.reward)
                if self.config.second_pass and retry and self.llm.available:
                    self._log(f"second pass: {len(retry)} tasks with {second_model}")
                    await self._pool(
                        retry, lambda t: self._attempt(t, second_model, "pass2", self.config.max_repairs)
                    )
                verify = [t for t in tasks if self._wants_verification(t)]
                verify.sort(key=lambda t: -t.reward)
                if verify and self.llm.available and second_model != self.config.model:
                    self._log(f"verification: {len(verify)} answers with {second_model}")
                    await self._pool(verify, lambda t: self._verify(t, second_model))
        finally:
            if self.llm is not None:
                await self.llm.aclose()
            await workflow.shutdown()
        return self._report()

    def _report(self) -> dict:
        records = list(self.records.values())
        summary = {
            "tasks": len(records),
            "answered": sum(r.status == "answered" for r in records),
            "answered_free": sum(r.status == "answered" and r.cost == 0 for r in records),
            "infeasible": sum(r.status == "infeasible" for r in records),
            "failed": sum(r.status == "failed" for r in records),
            "not_attempted": sum(r.status in ("pending", "skipped") for r in records),
            "reward_answered": sum(r.reward for r in records if r.status == "answered"),
            "tokens_spent": round(self.ledger.spent, 1),
            "llm_calls": self.ledger.calls,
            "llm_disabled_reason": self.llm.dead_reason if self.llm else "no LLM_API_KEY/LLM_BASE_URL",
            "estimated_tokens": round(sum(r.estimate for r in records), 1),
            "engines": self.engines,
            "elapsed_s": round(time.monotonic() - self.started, 1),
            "model": self.config.model,
            "second_model": self.config.second_model,
        }
        log_dir = self.config.log_dir
        if log_dir is not None:
            try:
                lines = []
                for record in records:
                    lines.append(
                        json.dumps(
                            {"record": asdict(record), "events": self.traces[record.id].events},
                            ensure_ascii=False,
                        )
                    )
                atomic_write(Path(log_dir) / "trajectories.jsonl", "\n".join(lines) + "\n")
                atomic_write(
                    Path(log_dir) / "summary.json",
                    json.dumps({"summary": summary, "tasks": [asdict(r) for r in records]}, ensure_ascii=False, indent=1),
                )
            except OSError as exc:
                self._log(f"cannot write logs: {exc}")
        return summary
