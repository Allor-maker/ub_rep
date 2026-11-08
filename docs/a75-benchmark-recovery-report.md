# Astra A7.5 — recovery of the 50-task run

2026-11-08. Source: `C:/VSCode/openJiuwen/live-test/tasks2`; updated baseline
`live-test/logs.txt`. Feature branch retained; no commit, push or merge.
The new experiment allowance is **50000 competition tokens = 500000 Units**.
All new subprocesses share one durable journal; earlier A7.4 and the supplied
baseline are reported separately and are never hidden as a new experiment.

**Final independent grader: 46/50 correct, 0 incorrect, 4 missing,
TAMPERED=0, AUDIT_ERRORS=0; reward 3321/3758.**

| Metric | Supplied baseline | One full recovery run |
|---|---:|---:|
| Correct | 5/50 | **46/50** |
| Reward | 482/3758 | **3321/3758** |
| Math | 2/18 | 15/18 |
| Search | 0/16 | **16/16** |
| Data Analysis | 3/16 | 15/16 |
| Design-free | 0/14 | **14/14** |
| Design-LLM | 5/36 | 32/36 |
| Inference HTTP | 55 | **35** |
| Competition tokens | 19289.7 | **10198.1** |
| Held Units / incorrect publications | 0 / 0 | **0 / 0** |

The full recovery used 47.1% fewer reported tokens than the supplied baseline.
**All new A7.5 experiments together: 139201 Units = 13920.1 competition tokens,
47 inference HTTP + 2 quota GETs = 49 HTTP, held=0.** Remaining authorized budget:
36079.9 competition tokens. No further experiment followed the user's final
instruction; no commit, push or merge was performed.

## Confirmed baseline and causes

The structured log confirms **50 loaded, 5 publications, 55 dispatched and 55
settled HTTP, 192897 Units = 19289.7 competition tokens, held=0**, not a transport
or accounting failure. Baseline score supplied with the benchmark: **5/50,
0 incorrect, reward 482/3758**. Exactly five original answer files were hashed
before work; experiments use fresh public-only copies and leave these files intact.

| Observed cause | Evidence and effect |
|---|---|
| 0/14 design-free | The free portfolio pass ran, but closed full-instruction grammar rejected depot/carton/inventory variants, changed CSV count/quantity wording and output suffix `without explanation`. The lowest-reward tasks subsequently lost paid scheduling eligibility as budget depleted. Design-free metadata is not an agent routing guarantee. |
| Source punctuation/discovery | `workspace/kb/.` failed the old directory boundary check; implicit local Markdown KB and `kb/ directory under workspace/` were not consistently selected. Empty evidence explains eligible Search model declines. |
| Nine source-limit skips | The prospective paid context admitted at most 3 files, whereas public corpora contain 4–16 short profiles. These were stopped before inference. |
| Nine incomplete replies | All are the second, Reviewer request after a completed Solver reply. Reported output equals cap in every case: eight 512, one 160. Raw provider finish strings were not retained, so cap exhaustion is strongly indicated; this is not evidence of a JSON parser defect. |
| Review cost | Inferring role from request order and route transitions gives 77235 Solver and **115662 Reviewer Units** (about 60% spent on review). Supported paraphrases did not bind to the existing independent semantics, so valid tool candidates still needed review. |
| Eight review rejections, three proposal/evidence errors | Exact old response/schema content cannot be reconstructed from this log. New diagnostics distinguish JSON, schema, source, computation/semantic validation, reviewer verdict and actual length finish. No specific lost correct answer is invented. |
| Budget routing | Reward-first unknown offers bought expensive/unverifiable reviews. The free pass itself was correctly prioritized; most resource deferrals were downstream consequences of grammar/source failures and review spend. |

The baseline's existing five submissions used review-based policy and have
`independent_semantics=false`; the recovery experiments require independent
semantics instead. No old validation checks or safety requirements were removed.

## Changes and scope

- `solvers/language.py`, `plans.py`: bounded compositional inventory, CSV
  aggregate/predicate, profile-relation and discrete-math recognition. Numeric
  parameters, identities, files and values come from input. Residual material
  words, additional numbers, unknown negations/alternatives and conflicting
  aggregate conditions refuse certification. Simple stock, birth/employer-city,
  row-count and unit-sum families gain deterministic routes.
- `solvers/sources.py`: sentence punctuation and explicit/implicit Markdown
  scopes are resolved against the public manifest only. Parse/source-selection
  caches are bounded and pure; mutable proposals, answers and validation are not
  cached. ToolCalls are reconstructed from immutable encodings.
- `tools/retrieval.py`, `tools/local.py`: scan at most 128 profiles / 65536 bytes,
  validate whole-corpus structure/injection/hashes, select explicit named seeds
  and at most four link levels / 16 profiles. Send at most 16384 bytes of relevant
  property excerpts, with original provenance; independently reopen originals
  for validation. Irrelevant injection and duplicate identities are not hidden.
- `orchestration/decision.py`: the host binds the complete allowed corpus, so
  model path hints cannot accidentally omit a chain source. Forbidden paths are
  rejected first; the independent verifier still reopens the complete original
  corpus. This changes source binding, not the Validator's full-corpus/evidence
  requirements. Three-field calls need no filenames; legacy allowed path hints
  remain accepted. An empty `field` with the terminal property in the last chain
  slot is normalized mechanically, without guessing an entity, fact or answer.
  Complete supported semantics bypass Reviewer; unsupported semantics can stop
  before HTTP in the recommended recovery policy. One opt-in Solver length
  retry increases cap to at most 2048 only if reported usage is exact, the entire
  next reservation fits the remaining attempt budget and time/steps remain.
  Unknown usage, non-length finishes, invalid JSON and a second truncation do
  not retry. Reviewer output is requested compactly.
- `tools/calculations.py`, `validation/{data_plan,decision,routed,instruction,local}.py`:
  exact same-column set membership, inclusive ranges/thresholds, combination and
  multiple-sum semantics. Independent Decimal/reader/graph or original-question
  computation remains blocking; CSV order rows are not assumed to be distinct
  orders, and unknown qualifications cannot be rescued by a Reviewer accept.
- `planning/decision.py`: in recovery mode, prioritize independently checkable
  work by reward / conservative reservation proxy, without inventing measured
  success probabilities. Per-attempt allocation, at most one length retry,
  deadlines and global A2 reservations bound unsuccessful spending.
- `orchestration/runtime.py`, `cli.py`, `solvers/composition.py`: preserve policy
  flags when creating the isolated run controller; expose discovered/processed,
  free/paid/publication counts, final skip reasons, category spend, truncations,
  Reviewer verdicts, validation checks and remaining Units.

Contracts v1, the Ledger, Gateway settlement, filesystem isolation and A5 atomic
no-clobber publication are unchanged. All orchestration uses openJiuwen 0.1.19.
`.env.example` documents the recovery opt-ins; legacy policy defaults remain
compatible. Run with **LLM_REQUIRE_INDEPENDENT=1, LLM_LENGTH_RETRY=1** as the
experiment harness does. These flags are configuration, not provider billing
attestations. The A7.4 report has a separate erratum for its old discarded
instance-level repair wrapper; its successful answers/accounting are unchanged.

## Local verification before paid requests

Zero-budget execution of all 50 public TaskSpecs: **14 independently verified
A5 publications: Math 6, Search 4, Data Analysis 4; zero HTTP, zero spent/held**.
The remaining 36 are classified as budget deferrals in this deliberate offline
run. This recovers all fourteen design-free cases without using their labels.

New integration cases cover changed operands/files/values, ambiguity and extra
conditions, exact set filters, reordered predicates, unrelated profiles,
long-document excerpts, injection, duplicate identities, corpus limits, free
publication before paid work, fresh parse objects and length retry/stop/unknown
accounting. Six cumulative-journal tests cover reconstruction, next reservation,
unknown/error/timeout, cap breach and secret redaction. Two legacy review fixtures
now use an unrecognized plain-text output phrase to retain their original review
coverage; the newly recognized `Return only the result` free behavior has its own
regression. Source/answer oracles are not read from the grader during development.

Full unittest before the final Search interface correction: **438 tests, OK,
6 skipped**, 262.452s (`final-unittest.log`). After that correction, **67 targeted
regressions passed**, Ruff passed, and Mypy passed for the changed controller/CLI;
the previous complete CI Mypy scope passed for 61 files. Four recorded failed
Search ToolCalls were replayed locally with zero HTTP: each executed against the
original corpus and matched the complete instruction after binding/normalization.
No additional full unittest was run after the final interface correction, in
accordance with the user's remaining Codex limit and priority for the full live run.
Host smoke passed (3/6 portable-reference answers, unsolved outputs absent, zero
budget). Six platform-specific Windows skips remain for Linux CI.

The old 20-task replay still produced 20 independent local publications with
MockTransport and held=0. Its old external `live-test/grade.py` is no longer
present, so that replay's external grading failed to start; no fresh externally
graded 20/20 claim is made here.

## Experiments and independent evaluation

All live calls used `qwen3.6-35b-a3b-noreason`, competition accounting, real
openJiuwen/Gateway/HTTP, output cap 512, independent-only policy and one affordable
length-retry option. **No live truncation, retry, Reviewer request, API accounting
error or unknown held occurred.** Every successful publication had
`independent_semantics=true` and all blocking checks passed.

| Experiment | Tasks / configuration | Inference HTTP | Units | Competition tokens | Verified publications |
|---|---|---:|---:|---:|---|
| `offline-diagnosis` | 50, budget 0, physical transport denied | 0 | 0 | 0 | 14 local independent |
| `offline-final` | 50, budget 0, final local families | 0 | 0 | 0 | 14/50, reward 190/3758, grader confirmed |
| `smoke-12` | 4 Math + 4 Search + 4 Data, local budget 10000 tokens | 12 | 37220 | 3722.0 | 8/12: Math 4, Data 4, Search 0 |
| `full-50` | 50, local budget **20000 tokens**, same as original baseline | 35 | 101981 | 10198.1 | **46/50, reward 3321/3758** |
| **All new paid runs** | one persistent series, 50000-token ceiling | **47** | **139201** | **13920.1** | **0 incorrect publications** |

The initial small Search calls exposed four actionable interface failures: two
completed lookups were rejected for supplying only visited paths, one omitted
the starting profile path, and one placed the final property in `chain` with an
empty `field`. The final host corpus binding/terminal-slot normalization fixed
these without changing the independent graph or evidence checker. No separate
paid Search retest was started: the user's final priority instruction was followed
with one full 50-task run. That run solved all 16 Search cases, including novel
birth/HQ, mentor/employer, year-specific summit, initiative-month and three-link
questions.

The private evaluator was extracted/executed only after local development and
checks, in `experiments/runs/a75/evaluator/`, outside every agent input/staging tree.
Its source and answer key were never inspected. The initial A7.3-derived exporter
serialized `task.json` with different whitespace. The JSON dictionaries were
identical, but the grader's byte-integrity check marked the exports tampered.
The final exporter preserves the exact original public manifest bytes after
asserting its five-key public schema and semantic equality.

Existing small/offline outputs were not edited. To evaluate those already-produced
answer bytes, separate **evaluation views** used original public input bytes and
copies of their A5 answers, after asserting JSON equality and all workspace-file
SHA256s. They scored 14/50 and 8/50 respectively, zero incorrect/tampered/audit errors.
These views are grader adapters, not additional agent publications or live runs.
The final 50-task run was graded **directly**, with no projection or input edit.

Per-category full-run expenditure:

| Category | Correct | Model tasks / HTTP | Units | Competition tokens |
|---|---:|---:|---:|---:|
| Math | 15/18 | 12 | 12221 | 1222.1 |
| Search | 16/16 | 12 | 28133 | 2813.3 |
| Data Analysis | 15/16 | 11 | 61627 | 6162.7 |
| **Total** | **46/50** | **35** | **101981** | **10198.1** |

The per-attempt totals, per-run totals and all durable per-request settlements
reconcile exactly. Provider quota checks allowed dispatch before each experiment.
No counter was reset between runs. The original five answer hashes were checked
unchanged after completion.

## Four remaining full-run skips

| Task | Observed cause | Safety result |
|---|---|---|
| `data_analysis_2007` | The revenue-definition guard does not recognize the equivalent phrase `multiply units by price`; independent semantics unavailable before dispatch. | 0 HTTP; no guessed/review-only publication. |
| `math_2012` | Solver proposed `9 * 9 * 8 * 7` for distinct-digit arrangements. | Original-question independent semantics failed; no publication. |
| `math_2015` | Solver's direct prime-sum answer was `79`. | Original-question independent semantics failed; no publication. |
| `math_2016` | Solver declined, incorrectly claiming the combination cannot be expressed using the supported arithmetic. | No publication or second paid request. |

The committee task was solved correctly in the initial small run; its original
[published answer remains preserved](../experiments/runs/a75/smoke-12/tasks/math_2016/answer.txt).
It was not copied into the full-run task tree or counted as a full-run success.
This illustrates model variability. These four gaps were documented, not repaired
with further experiments after the stop instruction.

## Saved results

- Final A5 outputs: `experiments/runs/a75/full-50/tasks/`.
- [Direct final external grade](../experiments/runs/a75/full-50-grade.log).
- [Machine-readable totals and all 50 task results](../experiments/runs/a75/results.json).
- [Per-task result / HTTP / Units / skip table](../experiments/runs/a75/task-results.md).
- [Persistent series journal](../experiments/runs/a75/journal.json).
- Each run retains `experiment.json` (configuration + production source SHA256),
  `summary.json` (safe request timings/counters and checks), raw local model/tool/
  validation diagnostics and `agent.log`. The initial failed-integrity grader
  output is retained alongside the corrected evaluation-view results.

The series is marked stopped by the agent; its cumulative counter is preserved.
Future manual runs, if requested, consume the same remaining allowance and must
use fresh labels. No background process is spending budget.

## Reproduction

The existing configured dotenv aliases are loaded privately by the harness.
Never delete `experiments/runs/a75/journal.json` to reset the allowance. A held
unknown, closed session or exclusive lock prevents another paid experiment.
Use a fresh label; existing run directories and original correct answers are
never removed or overwritten.

```powershell
Set-Location 'C:\VSCode\openJiuwen\nsu-budget-agent'
$env:PYTHONPATH = 'src;.'
$env:PYTHONIOENCODING = 'utf-8'
.\.venv-a1\Scripts\python.exe -m tests.benchmarks.a75 --mode offline --label offline-repeat
.\.venv-a1\Scripts\python.exe -m tests.benchmarks.a75 --mode live --label full-repeat --budget-tokens 20000 --output-cap 512
```

Each live invocation checks provider quota, writes an immutable experiment
configuration and production-source SHA256, then starts one normal CLI process.
Its local run budget is bounded by the remaining cumulative 500000-Unit allowance.
Raw local diagnostics are ignored by Git and ACL-restricted; no key or HTTP
authorization/header dump is recorded. Reported usage/estimates are not proof of
the provider invoice: unknown usage retains its reservation and stops the series.

This is a recovery evaluation on public inputs inspected during diagnosis, not a
fresh held-out generalization result. The language verifier has bounded domain
coverage; unrelated or ambiguous prose remains unsupported. No task IDs, answer
tables, gold or grader code occur in production logic.
