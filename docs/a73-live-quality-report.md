# A7.3 — diagnosis of 8/20 and controlled live retest

Date: 2026-11-08. Branch: `feature/a72-neuraldeep-production`.

**Verdict: CHANGES REQUIRED.** The authorized smoke produced one newly solved
LLM task with exact reported usage, but only 1/3 selected tasks passed. The
conditional full live run was therefore not started. Mock 20/20 is not evidence
of real Qwen quality or provider accounting guarantees.

## Evidence and isolation

Inputs: `C:\VSCode\openJiuwen\live-test\tasks`, exactly the requested 20 IDs.
Original stdout: `C:\VSCode\openJiuwen\live-test\logs.txt` (104533 bytes).
Original outputs graded in a separate process: **8 correct, 0 incorrect,
12 missing, reward 80/962**; free 8/8, LLM 0/12. The stdout confirms 8 saved,
2653 spent Units, 8324 held Units, 100000 limit Units, 22 SDK workflows,
31 component steps. Three attempted LLM role requests appear in route traces.
The old log has no per-request usage events: attempted is not independently
equivalent to HTTP dispatched, and precise per-task settlement is unavailable.

`tests/benchmarks/a73.py` exports TaskSpecs and their public artifacts into a
fresh task tree. No old answer.txt, answer key, expected/gold or grader is copied.
The normal `run_agent.py` runs as a separate process, with actual openJiuwen
workflows; only the offline gateway uses MockTransport and denies real transport.
The external grader runs after agent exit, outside the agent's task tree, with
no expected-answer display. Production code contains no benchmark IDs or answer
table. Original tasks, outputs and grader were not edited.

The configured dotenv file was loaded only into the live child environment,
mapping existing NEURALDEEP key/base URL aliases to LLM aliases. No key value was
printed. Model: `qwen3.6-35b-a3b-noreason`.

Windows sandbox traversal could not obtain a file identity for an ancestor of
the user TEMP directory. Harness copies and child TEMP were moved to the workspace;
the anchored filesystem checks were preserved. Two initial harness runs failed
locally before a usable replay; neither sent real HTTP.

## Original routes: all twelve paid-group tasks

All twelve passed FREE_ATTEMPT and reached UNSOLVED. Existing free grammar could
not fully bind these instructions; it did not guess partial answers. A4 performed
the entire free portfolio pass before paid scheduling. Highest paid reward was
math_1007, followed by data_analysis_1005. A2 blocks all further paid dispatch
while unknown usage is held, even with a positive nominal balance.

| Task | Free limitation | Old paid route / eligibility | HTTP / structured outcome | Old spent / held |
|---|---|---|---|---|
| math_1004 | Prime-sum semantics unsupported | deferred; no paid selection after hold | no attempted request; no candidate | 0 / 0 |
| math_1005 | LCM semantics unsupported | deferred after hold | no attempted request | 0 / 0 |
| math_1006 | Arithmetic-series semantics unsupported | deferred after hold | no attempted request | 0 / 0 |
| math_1007 | Distinct-digit arrangements unsupported | generic solve → tool → Reviewer → SKIP | 2 role requests; executable plan reached tool stage; no publish | not separately logged; likely 2653 / 0 |
| data_analysis_1002 | Full multi-filter revenue instruction unsupported | deferred after hold | no attempted request | 0 / 0 |
| data_analysis_1003 | Row-count/status instruction unsupported | deferred after hold | no attempted request | 0 / 0 |
| data_analysis_1004 | Full month/status/unit instruction unsupported | deferred after hold | no attempted request | 0 / 0 |
| data_analysis_1005 | Full month/status/revenue instruction unsupported | generic solve → SKIP | 1 role request, terminal workflow around 30s; response/usage not recorded | likely 0 / 8324; attribution inferred |
| search_1002 | Parent-organization chain unsupported | deferred after hold | no attempted request | 0 / 0 |
| search_1003 | Coordinator/year-specific property unsupported | deferred after hold | no attempted request | 0 / 0 |
| search_1004 | Mentor/employer chain unsupported | deferred after hold | no attempted request | 0 / 0 |
| search_1005 | Lab/coordinator/initiative property unsupported | deferred after hold | no attempted request | 0 / 0 |

Only two paid task IDs appear, and all spent/held belongs to those attempts.
Per-task attribution above is an inference from ordering/transitions, not a
reconstructed provider receipt. Deferred does not mean those tasks permanently
exhausted eligibility. They were prevented from running safely in this run.

## Reproduced defects and changes

1. **Source selection dropped workspace aliases.** A manifest path `sales.csv`
   did not match `workspace/sales.csv`; `workspace/kb/` did not select its Markdown
   sources. `solvers/sources.py` now resolves the explicit artifact-root prefix
   and scoped Markdown directories. Only manifest sources are eligible;
   unreferenced files and sensitive paths remain excluded. Tool/evidence aliases
   are canonicalized only when they resolve to that allowlist.
2. **Numeric Markdown property keys were rejected.** A property such as
   `2022 summit host city` invalidated a profile. Both executor and independent
   verifier now support bounded alphanumeric keys. Initiative-month results are
   checked against English month names.
3. **Independently verifiable direct proposals took unnecessary review routes.**
   `_known` now recognizes available complete math/data semantics and semantic
   equivalence of reordered AND filters/Markdown source sets. Validator still
   independently computes and checks the result. A wrong CSV total or wrong math
   answer cannot be rescued by Reviewer agreement.
4. **Missing independent discrete/multi-condition verification.** Added bounded
   parameterized prime sums, LCM, arithmetic series, arrangements, CSV predicates
   and explicit relational Markdown chains. Math computation differs from model
   expressions; CSV uses an independent Decimal reader; Markdown independently
   traverses original hash-bound documents. Whole supported text must be consumed.
   Unknown negations, qualifiers, conflicting aggregate clarifications, invalid
   months or unsupported prose do not receive certification.
5. **Timeouts and diagnostics were conflated.** CLI now exposes separate
   `--attempt-timeout-s` (default 60, range 1..300); competition HTTP timeout is
   `LLM_HTTP_TIMEOUT_S` (default 45, range 1..180), still bounded by remaining
   attempt/run deadlines. Global default remains 3300 seconds, max 3600. Explicit
   900-second benchmark deadline is independent. Gateway events contain request/
   attempt IDs, elapsed time, estimate/cap, reported counters and exact/unknown
   status; CLI contains per-attempt spend/hold/dispatch and fixed refusal codes.
   A post-smoke diagnostic-only fix maps fixed human-readable refusal reasons to
   safe codes; no raw Reviewer explanation or candidate is added to stdout.

New grammars are deliberately limited semantic families, not a universal natural
language parser. They contain wording patterns observed during diagnosis, but
take names, numbers, paths, filters and answers from input. Changed-parameter
tests prevent answer-table implementations. Generalization beyond these families
has not been demonstrated; this remains a merge limitation, not a 20-task claim
of broad reasoning ability.

## Budget and deadline conclusions

No evidence supports a Units conversion bug: one competition token is ten Units;
reported cost is input + 10 × output. Old deferred tasks had substantial nominal
balance, but **8324 unknown held Units** intentionally closed further paid
dispatch. Releasing that hold would be unsafe. No such release was implemented.
Existing reservation/cap negotiation and reward-first portfolio tests still pass,
including 100 tasks and reservation 13312; expensive work is requeued on temporary
allocation refusal. Independently verifiable routes need one mandatory model role;
unsupported routes retain mandatory review preflight.

The old paid workflow timing was approximately 15:25:46 to 15:26:16, consistent
with the previous 30-second attempt/HTTP settings. The global 600-second value
could not override them. The old log does **not** prove that a valid answer existed
at timeout, which exact timeout layer fired first, or what provider generation
continued afterward. Increasing limits is a configurable policy change, not proof
of fixing a lost candidate. The new live smoke had no timeout or unknown usage.

A5 still awaits publication after acceptance before cleanup/next task, even after
attempt deadline. There is no fake five-second publication timeout. A blocked
kernel fsync can still prevent bounded completion; Python cannot safely interrupt
it. Crash/no-clobber/deadline tests remain part of the full suite. No background
publication or cleanup race was introduced.

## Offline results and validation

| Run | Correct / incorrect / missing | Reward | HTTP | Spent / held Units |
|---|---|---|---|---|
| Original real outputs (external grader) | 8 / 0 / 12 | 80/962 | 3 attempted roles; dispatch not separately logged | 2653 / 8324 |
| Current no-key host fallback, full public set | 8 / 0 / 12 | 80/962 | 0 | 0 / 0 |
| Current 20-task MockTransport replay | 20 / 0 / 0 | 962/962 | 12 mock | 7200 / 0 |
| Current selected 3-task mock, BUDGET=1500 | 3 / 0 / 0 selected | 119/962 | 3 mock | 1800 / 0 |

There was no historical mock run of these 20 tasks captured before the changes;
the original live baseline and current mock are different experiments. Mock
proposals are valid constructions from public inputs, not samples of Qwen.

Artifacts (under `experiments/runs/`, relative to project root):
`astra-a73-offline-d9q_tuvq/{summary.json,grade.log,agent.log}` (final full replay),
`astra-a73-offline-hi1mwf0w/` (earlier full replay),
`astra-a73-offline-x_pck61c/` (1500-token selected replay).

Regressions: `tests/integration/test_a73_quality.py` covers source scope/secrets,
numeric properties, direct answers, complete conditions and conflicting clauses,
changed math/CSV/search parameters, canonical tool paths and filter ordering,
wrong answers, independent local_math checks and content-free usage events.
`tests/unit/test_cli_deadline.py` checks independent CLI limits and safe codes.
The first regression run reproduced source/grammar/review failures (one failure,
five errors); later a conflicting aggregate regression also failed before its fix.

Checks: targeted integration/CLI/production tests 22 passed, plus the final
three CLI tests; final full unittest **363 tests, OK, 6 skipped**, 214.530s,
including the last diagnostic-only change; Ruff check/format 97 files;
Mypy CI scope 59 files; host smoke 3/6 portable and 8/20 public, zero-budget
fallback and unsolved output absence verified. An exploratory broader Mypy scope
also included the SDK probe and encountered its two untyped SDK imports; the
documented CI scope passes without suppressing those findings in code.

Final full log: `%TEMP%/a73-final-unittest.log`; targeted log:
`%TEMP%/a73-target.log`. Mypy and both host smokes were rerun successfully after
the final CLI change. `git diff --check` passes. Skipped Windows checks remain
the responsibility of the existing Linux CI; no Linux/Docker run is claimed.

## Authorized live smoke

Exactly one smoke, fresh copy: `experiments/runs/astra-a73-smoke-e2t13fv1/`.
Local BUDGET=1500 = 15000 Units; run deadline 240s, attempt 60s, HTTP 45s;
requested output cap 512, review cap negotiated to 383. It completed in about
nine seconds including startup and grading. All four reported usages settled;
**spent 7090 Units, held 0**. No uncontrolled retry or unknown request occurred.

| Task | Route | Result | HTTP | Spent Units |
|---|---|---|---|---|
| search_1002 | solve → tool → SKIP | invalid_proposal_or_evidence; no publication | 1 | 2356 |
| data_analysis_1002 | solve → tool → review → SKIP | rejected/unresolved after bounded strategy; no publication | 2 | 4056 |
| math_1004 | solve → local tool → Validator → A5 | correct answer.txt, independent semantics | 1 | 678 |

Reported input/output counters by call: search 716/164; data Solver 1050/49,
data Reviewer 996/152; math 338/34. Their weighted sum is 7090. Per-call elapsed
gateway times were approximately 1.250s, 0.453s, 2.016s and 0.594s. The external
grader reports **1 correct, 0 incorrect, 2 missing among selected tasks**,
reward **34/962** (selected reward denominator 119). Its full-set missing count
is 19 because seventeen tasks were intentionally not in this smoke.

Raw model responses were not retained. Therefore the exact bad Markdown arguments,
CSV filters/candidate and Reviewer explanation cannot be reconstructed. The
transitions prove structured tool plans were accepted into execution, not that
their semantics or execution results were valid. Do not label these failures as
JSON parsing errors, model arithmetic errors, or provider bugs without evidence.
The math task establishes the real Qwen → A7.1 → independent Validator → answer.txt
path for a task unavailable to the free solver. It does not establish 20-task
live coverage. No full live run or second smoke was made.

## Remaining risks and next action

- Two of three real source-driven tasks remain unsolved. Add bounded diagnostic
  classifications for tool-result/schema/semantic mismatch, then reproduce the
  observed proposal forms offline when evidence becomes available. Do not
  loosen validation or silently substitute a guessed plan to inflate coverage.
- Closed-family coverage and three selected real tasks are insufficient evidence
  for unseen contest prose. Broader varied paraphrase/negative-condition fixtures
  are needed before a merge-quality claim.
- Reported usage is not a provider billing attestation. Input estimate/framing,
  output caps and undisclosed reasoning remain competition-mode risks. Local
  BUDGET is not a guaranteed upper bound on the provider invoice; strict mode
  still needs a reviewed compatibility adapter and offline fallback needs no key.
- Unknown usage deliberately stops paid work. In-memory accounting cannot reconcile
  a process kill; no automatic paid restart with the same quota is safe.
- Filesystem/kernel I/O may block indefinitely after acceptance. Publication waits
  preserve atomicity and prevent cleanup races, not a hard process-duration bound.

Before another real attempt, review the limited semantic additions and capture
safe structured failure metadata with offline tests. A new live allowance is
required: the authorized smoke was used and the full-run prerequisite failed.
No commit, push or merge was performed.
