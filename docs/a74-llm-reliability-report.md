# A7.4 — extended LLM reliability sprint

2026-11-08, `feature/a72-neuraldeep-production`. This report covers only the
new experiments authorized for **25 HTTP / 10000 competition weighted tokens
(100000 Units) cumulatively**. Earlier A7.3 usage is not charged to this new
allowance. No commit, push or merge. Existing uncommitted A7.2/A7.3 changes were
preserved; Contracts v1, Gateway accounting and A5 publication were not changed.
The independent arithmetic checker gains the same bounded mathematical operation
through a separate implementation; no mandatory checks were removed or weakened.

**Verdict: LLM CORE WORKING on the eight tested tasks after the documented fixes.**
2 Math, 3 Search and 3 Data Analysis tasks have correct real-provider answers,
independent validation and A5 `answer.txt` publication. **Zero incorrect
publications, zero unknown usage/held.** Paid experiments were stopped after all
three categories had multiple distinct successes.

Total: **17 inference HTTP, 36611 reported Units = 3661.1 weighted tokens**, held=0.
Including 2 quota GETs and 3 conservatively counted public documentation attempts,
the session counted **22/25 HTTP**. The unused 63389 Units were not spent.

## Diagnosis, with limits on historical conclusions

A7.3 did not retain raw responses. Its exact Search/CSV ToolCalls cannot be
reconstructed. The following are **new observed failures**, not invented
explanations of those missing historical responses.

The original public Search and CSV inputs were exported into fresh copies first.
The initial diagnostic used the unchanged Solver prompt. Real endpoint/model:
configured NeuralDeep, `qwen3.6-35b-a3b-noreason`; actual `AsyncHTTPTransport`,
real openJiuwen **0.1.19**, A2/A3/A4/A5. The unrelated parent `my_env` contains
0.1.20.post1; its bootstrap was rejected locally before inference. All measured
agent experiments use `.venv-a1` and 0.1.19.

| Experiment | Actual model intent / tool execution | Outcome |
|---|---|---|
| `diagnosis-sdk019`, Search | `entity="Cedar Meridian Group", chain=[], field="Founded", paths=[terminal company file]` → `1994`, one evidence file | Reviewer accepted; independent full-chain evidence check rejected. No answer published. |
| `diagnosis-sdk019`, CSV | `product_sum(units,price)`; product=widget, region=west, status=paid, all `eq` string predicates | `106`, independently verified and externally graded correct. The old CSV failure did not reproduce. |
| `repaired`, Search | Original person, all three files, `chain=["Employer"], field="Parent organization"` | Returned intermediate company name. `instruction_plan_mismatch`; no Reviewer/publication. |
| `search-final-field`, Search | Correct two-link chain and `Founded`, but only two files, omitting final property's source | Tool `invalid_input`; `tool_execution_failed`; no publication. |
| `search-schema`, Search | Original person, full corpus, both relation keys, final property | Correct independently checked year published. |
| `source-holdout`, CSV period | Sum units with product/region/status, but **omitted month bounds** → `22` | `instruction_plan_mismatch`; no Reviewer/publication. |
| `data-period-repair` | Same task after generic contiguous-number/range guidance; both inclusive month bounds | Correct independently checked answer published. |
| `math-holdout`, LCM | `lcm(6, lcm(8, 9))` | Function calls unavailable, tool `invalid_input`; no publication. |
| `math-syntax-repair` | Same function call despite prose describing allowed operators | Failed closed; no publication. |
| `math-explicit-repair` | Same function call with numeric expression schema; the intended harness feedback was not preserved by runtime controller cloning (see erratum below) | Failed closed; no publication. |
| `math-bounded-lcm` | `lcm(6, 8, 9)` through the new bounded mathematical operation | `72`, independently verified against the original question and externally graded correct; A5 published. |

The diagnostic confirms that the model received the real Markdown documents and
CSV contents. There was no missing workspace alias or parser rejection of a
valid Search plan in these new samples. Failures were incomplete instruction
binding/source corpus and an unavailable expression operation. Correct CSV
arithmetic was not lost in serialization. Reviewer agreement could not override
the independent evidence failure.

## Changes

`orchestration/decision.py` now describes starting entity, ordered relation
traversal, final property and complete source corpus separately. Examples use
unrelated generic entities/relations, not benchmark instructions or answers.
Tool-specific prose is selected by available artifact suffixes, not benchmark IDs
or exact instructions. An intermediate all-tools prompt exceeded a 6500-Unit
small-budget Math fixture; scoping the prose restored that route without raising
its budget or changing Gateway negotiation. Shared source/injection rules remain
present for every task.
Every Solver request includes actual argument schemas derived only from the
public manifest and hash-bound CSV headers. Markdown paths have an enumerated,
unique full-corpus set; entity/chain/field remain model choices. CSV schemas
specify aggregates, string filter values and inclusive operators. The math
schema describes numeric arithmetic and the single bounded `lcm` operation.

Only after three observed real LCM failures, `tools/calculations.py` adds an
explicit AST case for `lcm`: 2..16 positive integral operands, each <=1000000,
within existing expression length/node/depth/result bounds. It computes Euclidean
LCM using integers. `validation/certification.py` independently uses Decimal
operand checks and the standard-library LCM, and the existing full-question
verifier still computes the required answer from the original question. Wrong
LCM operands cannot be rescued by Reviewer agreement. No `eval`, imports,
attribute dispatch, keyword/star arguments or arbitrary named functions are
enabled. This is a restricted mathematical operation, not Python execution.

Generic guidance covers AND conditions, contiguous numeric lists as inclusive
intervals, and declining unexpressible noncontiguous conditions. It never
silently substitutes missing filters or repairs a chain using a known answer.
Supported but mismatched tool plans stop with `instruction_plan_mismatch`, rather
than purchasing a Reviewer that cannot establish their correctness. Failed tools
stop with `tool_execution_failed`. The independent Validator is unchanged.

`tests/benchmarks/a74.py` is an explicitly invoked diagnostic harness. It uses
the ordinary CLI, real Gateway/SDK and original immutable instructions. A
separate, single-use `numeric_expression_only` repair option supplies the exact
safe syntax error, no answer or proposed numeric operands. It is diagnostic
orchestration, not an automatic production repair loop. Normal startup has no
raw diagnostic logging and no repair environment variable.

## Cumulative budget and isolation

One fsynced, atomically replaced journal spans all child processes:
`experiments/runs/a74/journal.json`. An exclusive session lock prevents concurrent
experiments. Before each inference the transport persists its HTTP count and
estimated full input/output reservation. Unknown outcome/usage leaves a durable
held amount and prohibits further inference. Exact reported counters settle
`input + 10 * output`; no cache discount. Reported overrun halts subsequent work.
Tests reconstruct the transport to prove counters do not reset between runs.
Input-bound or output-cap breaches also halt the durable session even when their
combined cost still fits the total reservation.

Initial `GET /limits` returned key=ok, can_request=true, session remaining=9996,
week remaining=999978. Its counters are provider quota data, **not a token billing
attestation**. Quota requests are conservatively included in the 25 HTTP ceiling;
they consume no inference Units. Three public documentation fetch attempts are
also conservatively counted, including the browser fetch with unsupported
Markdown content type. See [provider Limits API](https://neuraldeep.ru/docs/limits-api.md).
Client reservations and `max_tokens` do not guarantee a provider invoice ceiling.
Reported usage never became unknown in these measured live experiments.

Only TaskSpecs/public artifacts enter child task trees. Grader/answer keys are
not copied or supplied to the agent. The external grader runs in a separate
process after each agent exit. Original inputs/outputs and grader are unchanged.
Diagnostics contain no headers, credentials, URLs with secrets or environment
dumps; model content is additionally redacted against the loaded key. Raw local
files are Git-ignored under `experiments/runs/a74/`, with inheritance disabled
and access granted to the current Windows identity. No raw output was committed.

## Offline verification

`test_a74_reliability.py`: **45 SDK scenarios**: 12 changed Search corpora,
12 changed CSV files/values/condition orders, 13 malformed or incomplete plans,
4 month-range positive/negative cases, and 4 math syntax/computation cases.
They check workspace aliases, names versus filenames, relation/source omissions,
wrong columns/aggregate/operators/values/types, traversal, fake Reviewer accept,
independent publication and no extra HTTP. Existing A7.1–A7.3 regressions retain
wrong-result/evidence, source injection, forged tool output, unknown usage,
deadline, budget, crash and single-publication coverage. Two bounded-LCM unit
tests additionally cover four positive expressions and eleven prohibited operand/
call forms in both implementations. Six new journal tests
cover cumulative reservation, restart, HTTP cap, unknown usage and timeout.

The longer prompt exposed an obsolete overrun fixture: 4000 reported input
tokens no longer exceeded the input estimate. Mock input was increased to 40000;
the regression now asserts that its reported usage actually exceeds the current
input estimate or output cap. Budget enforcement was not changed to satisfy it.
An intermediate full run failed these fixtures; subsequent focused checks pass.
The later small-budget prompt regression was fixed in production prompt scoping,
not by increasing the test budget. Intermediate failed full logs are retained.

Final checks: **416 unittest tests, OK, 6 skipped**, 202.234 seconds, in
`experiments/runs/a74/scoped-final-unittest.log`. The six Windows skips concern
existing platform-specific checks; no Linux/Docker run is claimed. Ruff lint and
format pass (101 files); Mypy CI scope passes (59 files); `git diff --check` passes.
Host smoke: 3/6 portable public-reference answers, unsolved files absent,
zero budget/HTTP. Final full public MockTransport replay: **20/20, reward 962/962,
zero incorrect**, 12 mock requests, 7200 mock Units, held=0. All eight free public
tasks remain solved. Mock Units/requests are excluded from the real session ledger.
Replay: `experiments/runs/astra-a73-offline-c6ws93j0/`; host log:
`experiments/runs/a74/scoped-host-smoke.log`.

## Every live task and publication

HTTP and Units below include **all new failed attempts and repairs**, not only
the successful attempt. Answer links point to real A5 publications in fresh copies.
Every linked answer was graded correct by the external grader.

| Task | Final answer.txt | Inference HTTP | Total reported Units |
|---|---|---:|---:|
| `math_1005` | [72](../experiments/runs/a74/math-bounded-lcm/tasks/math_1005/answer.txt) | 4 | 4315 |
| `math_1006` | [216](../experiments/runs/a74/math-holdout/tasks/math_1006/answer.txt) | 1 | 1392 |
| `search_1002` | [1994](../experiments/runs/a74/search-schema/tasks/search_1002/answer.txt) | 5 | 10014 |
| `search_1003` | [Porto](../experiments/runs/a74/source-holdout/tasks/search_1003/answer.txt) | 1 | 2542 |
| `search_1004` | [2003](../experiments/runs/a74/source-holdout/tasks/search_1004/answer.txt) | 1 | 2413 |
| `data_analysis_1002` | [106](../experiments/runs/a74/repaired/tasks/data_analysis_1002/answer.txt) | 2 | 5355 |
| `data_analysis_1003` | [2](../experiments/runs/a74/source-holdout/tasks/data_analysis_1003/answer.txt) | 1 | 3282 |
| `data_analysis_1004` | [12](../experiments/runs/a74/data-period-repair/tasks/data_analysis_1004/answer.txt) | 2 | 7298 |
| **Total** | **8 different tasks correct** | **17** | **36611** |

Nine correct answer copies were published because the original CSV was also
included in the initial two-task retest. All other successes are distinct tasks.
All successful traces have `independent_semantics=true`, with blocking format,
paths, protected-files, snapshot and `llm_semantics` checks passed. No successful
route depended on Reviewer agreement; the sole Reviewer call was the rejected
initial Search diagnostic. Failed/intermediate candidates were never published.

This is **eventual 8/8 across diagnostic/fix versions**, not an 8/8 measurement
of a single frozen prompt. First attempts on these eight tasks solved 5/8.
Search coordinator/city and mentor/employer questions both passed at first
exposure after the original chain fix; CSV row-count and Math seating-series
formulations likewise passed without a task-specific correction. These provide
new formulations in all three categories. The period and LCM failures were
preserved and used for the subsequent fixes above.

## Every experiment and cumulative settlement

Each row has held=0. HTTP here means inference only.

| Run directory under `experiments/runs/a74/` | HTTP | Units | Cumulative Units |
|---|---:|---:|---:|
| `diagnosis-sdk019` | 3 | 6170 | 6170 |
| `repaired` | 2 | 4676 | 10846 |
| `search-final-field` | 1 | 2078 | 12924 |
| `search-schema` | 1 | 2445 | 15369 |
| `source-holdout` | 4 | 11555 | 26924 |
| `data-period-repair` | 1 | 3980 | 30904 |
| `math-holdout` | 2 | 2501 | 33405 |
| `math-syntax-repair` | 1 | 1188 | 34593 |
| `math-explicit-repair` | 1 | 1228 | 35821 |
| `math-bounded-lcm` | 1 | 790 | 36611 |

Per-request reported counters, model, timing and attempt IDs are in each
`summary.json`; actual proposals/tool results/checks are in local `model.jsonl`,
`tools.jsonl`, `validation.jsonl`. Machine-readable reconciled totals and answer
SHA256s are in [results.json](../experiments/runs/a74/results.json), and the
durable [journal](../experiments/runs/a74/journal.json) includes all non-inference
reads. Summed per-attempt, per-run and per-request Units agree exactly.

Final provider quota read: can_request=true, session used=21/remaining=9979,
week used=39/remaining=999961. Both quota counters increased by exactly 17 from
the initial read, consistent with the 17 inference requests. This is request
quota corroboration, not monetary/token billing reconciliation. All 17 usage
records settled unambiguously under competition policy. The local journal is
now deliberately closed (`quota_ok=false`, stop reason
`all_three_categories_confirmed_on_multiple_tasks`), even though provider quota
and authorized budget remain. No further paid request was made.

This evidence concerns bounded supported semantic families, not arbitrary natural
language or a guaranteed 20/20 live benchmark. Prose-only syntax guidance and one
explicit repair did not fix LCM; bounded operation support did. Schemas are supplied in messages, not enforced by provider
constrained decoding; malformed model plans remain possible and must fail closed.

## A7.5 inspection erratum

Runtime constructs a fresh DecisionController for run isolation. The A7.4
diagnostic's instance-level `make_request` replacement was therefore discarded.
The claim that `math-explicit-repair` supplied error feedback to Qwen was not
established and is corrected here. Its actual schema/proposal, HTTP and 1228 Units
remain recorded; no publication occurred. The later bounded-LCM success, all eight
graded answers, total 36611 Units and zero incorrect publications are unaffected.
A7.5 explicitly preserves its constructor policy options during controller
cloning and tests their effect through the actual runtime.
