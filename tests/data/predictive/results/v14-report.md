# JARVIS v14 — Behavioral graph stage report

## ARCHITECTURAL FAILURE ANALYSIS

Baseline: `b00a4813278699735d77c41a593cea99790ff8f7`. Historical v12/v13
holdouts were used for diagnosis, not as a new promotion population.
The before-code audit is preserved in `v14-architecture-diagnosis-before.md`.

Three independent losses were reproduced before runtime changes:

1. `_line_target` existence suppressed v13 symbol recovery even when the
   location did not connect the reported API to an implementation.
2. Callable resolution omitted qualified imports/method owners, while the
   causal builder could assign an unrelated same-leaf callable to a dynamic
   receiver. The latter is a false structural relation, not a ranking issue.
3. Return/condition-based seeding did not carry void attribute updates,
   subscription assignments, yields, or exception-handler clauses as typed
   behavioral operations. A definition in the ledger cannot prove these paths.

The new external development has two independent defects, not multiple
wordings counted as extra defects. Pytest #7432 has a report-state mutation
with an incorrect branch/configuration interaction; Sphinx #8713 has a
configuration-dependent docstring formatting behavior delegated through a
class API. A resolved operation still does not describe the violated contract.
The correct candidate is absent in the completed final Pytest live case;
there is therefore no valid conditional selector-accuracy denominator.

## API-TO-IMPLEMENTATION GRAPH

`behavioral.py` adds immutable callable, implementation-edge, effect and
unresolved-dispatch records, with stable snapshot IDs, owners, source ranges,
resolution methods and evidence IDs. Support states remain separate:
`AST_DEMONSTRATED`, `STATICALLY_RESOLVED`, `INFERRED_FROM_OBSERVATION`,
`UNRESOLVED`. No observed/inferred edge is silently upgraded.

Resolution supports unique local calls, imported aliases/modules, relative
imports and constrained lexical `self.method` calls. It rejects decorated,
shadowed, inherited/dynamic or unresolved receivers rather than matching a
callable solely by its final name. This is a static-snapshot model, not a
universal guarantee about monkeypatching or runtime dispatch. Class
constructors, complex decorators and general polymorphism are not covered.

The existing index is reused. One bounded traversal has depth 2, at most 2
additional files and 16 symbol contexts, a 48,000-byte context ceiling,
240 lines per symbol, and a 4,000,000-byte source-parse budget. An already
oversized original context is diagnosed; new context cannot increase it.
Original windows/anchors are retained. Genuine traceback flows do not
materialize new windows unless the first slice produces no roots. Behavioral
API exploration can proceed despite an existing inadequate line anchor.

JSON schema 6 is retained with additive `behavioral_graph_schema_version=1`
and stage timings. Existing CLI commands, policy, scoring weights, v7 recovery
budgets and v11 fail-closed sandbox are preserved. One Qwen reasoning call
remains the normal maximum; the graph is constructed deterministically.

## BEHAVIORAL EFFECT MODEL

AST operations represented: `RETURNS_VALUE`, `WRITES_ATTRIBUTE`,
`MUTATES_CONTAINER` (subscription assignments), `YIELDS_VALUE`,
`RAISES_EXCEPTION`, `HANDLES_EXCEPTION`. The effect means an operation is
present in source, not that execution occurred or the operation is wrong.
Handler presence does not prove a dynamic exception/handler match.

Ledger binding requires the exact owner/site/kind/call expression; there is
no same-file evidence fallback for new graph edges. New causal graph nodes
and edges are `CALLABLE`, `BEHAVIORAL_EFFECT`, `IMPLEMENTATION_CALL`,
`HAS_EFFECT`. Missing atoms remain observable as `MISSING_LEDGER_RELATION`.
Old legacy evidence fallback is not silently presented as this stricter
validation. No fabricated `effect -> symptom` or `EXCEPTION_SITE` is added.

Remaining gaps: demonstrated violated behavioral contracts, attribute
write-to-read lineage, iteration consumption, configuration effects and
API-class construction/dispatch. Reachability-only nodes cannot generate
a strong root recommendation. `contract_supported=false` is explicit.

## CAUSAL REPRESENTABILITY BEFORE/AFTER

Populations and proxies are deliberately not conflated:

| Measurement | Frozen v13 | v14 | Eligible population |
| --- | ---: | ---: | --- |
| Selected implementation symbol proxy | 8/10 | 10/10 | Controlled families with known implementation |
| Actual API/implementation path control | Not instrumented | 10/10 | Same controlled positive path families |
| AST effect control | Not instrumented | 11/11 | Controlled families with effect oracle |
| Qualified callee-return edge precision, OFF/ON ablation | 3/5 | 4/4 | Hand-adjudicated call-identity controls |
| External development symbol in ledger, offline | 1/2 | 2/2 | Two independent defects |
| External development localized root candidate, offline | 1/2 | 1/2 | Same defects; location proxy, not semantic proof |
| Demonstrated external defect relation | Not measured | Not adjudicated | No eligible semantic-contract population |

Four qualified return-producer edges are correct; two false legacy edges
are removed. Positive/negative rename, wording, comment, receiver, byte/file
budget and counterfactual controls are unit tests, not invented live percentages.
The controls have no violated-contract oracle; they do not establish root
diagnosis accuracy. Earlier prototype reports are retained as progress traces.
Their implementation-presence proxy must not be compared as API-path recall.
The final external evaluator requires an independently known affected API
and a graph path; an unknown API is undefined, not an automatic success.
The final offline rerun (`v14-external-development-offline-stage-gate.json`)
has API-path recall 1/2 and effect availability 2/2, while localized candidate
recall remains 1/2. The class entry point in Sphinx is not resolved into its
implementation. This is transport improvement without demonstrated causal
candidate-generation improvement.

## ORACLE INTERVENTIONS

Two development cases, offline evaluator only, no gold diff in the runtime:

| Mode | Observation | Limitation |
| --- | --- | --- |
| A — normal | 1/2 localized root candidates | Location proxy only |
| B — known affected API | 1/2 localized root candidates | No measured improvement |
| C — independently known path | 0/2 adjudicated | Upstream correction does not establish full dispatch path |
| D — AST-backed known effect | 2/2 effect observations; 1/2 localized roots | An operation is not a violated contract |
| E — grounded candidate | Available in one case; not representable in the other | Selector not run in this offline study |

Consequently, this study does not claim a complete A–E selection experiment.
It does show that transport of an effect alone is insufficient. C and E
limitations are preserved explicitly instead of supplying guessed oracle roots.

## EXTERNAL DEVELOPMENT

The frozen engine baseline is structural/offline, not an invented live BEFORE.
Final live development evaluates 1/2 independent defects; one Sphinx case is
`NOT_EVALUATED_TIMEOUT` at 240 seconds. Among completed defects:
file 1/1, ledger symbol 1/1, candidate 0/1, root 0/1, repair pair 0/1,
recommendation precision 0/1, coverage 1/1, false abstention 0/1.
Conditional selection accuracy is undefined (0 eligible correct-candidate cases).
API-path/effect availability are each 1/1; defect representability is undefined.
These tiny denominators are not evidence of broad external generalization.

## SEALED EXTERNAL HOLDOUT

Four new independent defects from Pytest/Sphinx: two development and two
held out, one original formulation each. Corpus frozen before runtime edits,
at `2026-10-10T22:37:46.811176+00:00`, SHA-256:
`ca35ac2aeef385022a3b6e6a6f6224a7dd9bfdf094180b3a0beb02347d3321b9`.
Commit/source hashes, upstream references and adjudications are in
`external_v14/cases.json` and `manifest.json`; Python-only snapshots are
read-only evaluation inputs, not imported modules. Thresholds were sealed
at 75% implementation/candidate recall and root validity, 100% resolved-call
precision. No threshold reduction or post-holdout runtime tuning is allowed.

The two held-out defects do not have independently adjudicated public API
identities; API-path recall therefore has an empty eligible population and
must remain undefined. A corrected helper location is not substituted for
the affected API. The sample is purposive and too small for production claims.

One-shot results are recorded in `v14-external-holdout-one-shot.json`.
Exactly one invocation, engine byte hashes unchanged since the final runtime
freeze (`9c8f7811b6232973bd75234c2c562ceb184030cd`,
`2026-10-10T23:10:08.977698+00:00`); no post-holdout tuning.

| Held-out result | Numerator/denominator | Population |
| --- | ---: | --- |
| Completed | 1/2 | Independent defects / one formulation each |
| Timeout | 1/2 | Same population; 10451 stopped at 240 s |
| File retrieved / symbol in ledger | 1/1 each | Completed defect 8721 |
| Correct root candidate | 0/1 | Completed defect |
| Root / repair-pair validity | 0/1 each | Completed defect |
| Recommendation coverage | 0/1 | Completed defect |
| False abstention | 1/1 | Completed positive defect |
| Recommendation precision | Undefined (0 eligible) | No recommendation |
| API-path recall | Undefined (0 eligible) | No independently known affected API |
| Behavioral effect availability | 0/1 | Completed defect |
| Defect-relation representability | Not adjudicated | No eligible contract oracle |

8721 exposes a generator implementation but has no usable reported API seed;
the correct symbol in the ledger is not enough to construct a supported
root. Failure codes: `CORRECT_ROOT_CANDIDATE_MISSING`, `FALSE_ABSTENTION`.
10451 is operationally incomplete, not a causal error or abstention. For the
0/1 root estimate, the descriptive Wilson interval is [0, 79.3%]; this does
not remedy the tiny purposive population or the missing timed-out result.

## SYNTHETIC REGRESSION

Offline v10 structural regression: oracle root candidate recall 172/172,
argument-slot identity 102/102, import SCC recall 10/10. This is independent
from the external corpus. The live regression is a stratified 14-case sample,
not a repeat of all historical live holdouts. Original v10 results are retained.
Live structural oracle results: root 14/14, repair pair 14/14, top-1 14/14,
recommendation precision 12/12; two appropriate negative-control abstentions.
There were 12 model calls across 14 cases and no extra comparison calls.
The retained legacy lexical evaluation says 2/14 passed and reports 12
`WRONG_ROOT_CAUSE` codes; it has hypothesis-hit 0 and undefined causal-hit.
Those obsolete labels are not substituted for v10's structural oracle, and
their disagreement is disclosed rather than deleting historical diagnostics.

## LATENCY AND RESOURCE BUDGET

Separate observations cover index, retrieval, initial context, symbol
resolution, implementation graph, ledger, causal slice, recovery, Qwen and
ranking. Inclusive context/recovery timings overlap child stages; do not sum
them as independent work. Observed timing fields are excluded from exact
decision-determinism comparisons, but preserved for profiling.
Zero phase values in structural-only external reports are placeholders for
unmeasured timing, not proof of instantaneous stages. Completed live profiles
are the latency evidence. A killed worker does not provide a complete
per-stage trace, so its timeout cannot be attributed confidently to the model,
index or graph stage; this is an observability limitation.

Final completed Pytest diagnostic: index 5.797 s, retrieval 0.285 s,
implementation graph 0.192 s, ledger 0.557 s, slice 0.970 s,
Qwen request 15.120 s, ranking 0.0003 s. The Sphinx 240-second censored
case is not assigned a latency or causal error score. Colocated tests and
the single-slot local model can affect wall time and queueing.
Synthetic live Qwen mean: 47.580 s across 12 requests (p95 78.026 s), above
the 45-second operational target. Mean full synthetic case wall time:
41.203 s across 14 cases, including negative controls without Qwen calls.
Estimated prompt/completion tokens per request: 3257/419. These are estimated
tokens; no time-to-first-token measurement is available.
Completed holdout 8721 required no Qwen call. Its service profile has index
15.505 s, retrieval 0.861 s, context 1.156 s, ledger 0.732 s, slice 9.509 s
and recovery 2.683 s. Zero model requests means model latency is undefined,
not 0 seconds per request. 10451 remains censored at 240 seconds.

Cached per-line UTF-8 source slicing preserves `ast.get_source_segment`
outputs (including Unicode/multiline/formfeed controls). Microbenchmark:
200 segments from a 208,382-byte source, old 0.0471 s versus cached
0.000281 s. This is not an end-to-end speed claim.

Static phase checks are cooperative (60 s/stage, 240 s/diagnosis), with Qwen
HTTP timeout 180 s. External evaluation additionally terminates its own
read-only worker after 240 s and samples Windows working set with a
1024 MiB limit. This is not a hard memory allocation ceiling or a project
execution sandbox. Timeout/resource cases are excluded from quality and
abstention denominators and block promotion; they are never causal failures.

## SAFETY AND TESTS

Only trusted JARVIS unit tests/evaluator workers execute on the host. External
source is indexed and parsed as data: no third-party tests, installers,
imports, patches or simulation. Docker executable remains unavailable;
all sandbox capability checks are false and fail closed. No installation or
privileged host configuration was attempted.

The generic scoring weights, candidate policy, recovery budget and sandbox
implementation remain frozen. A narrow regression fix prevents new callable
and effect nodes from replacing parameter target identities; no score
weights were changed. All new graph edges require exact ledger provenance.

Final complete suite: **1582 passed, 3 skipped**, one existing
`VOICE_TTS_SPEED` warning, 214.57 s (the earlier runtime validation also
passed, in 166.41 s). Predictive-only suite: **341 passed,
2 skipped**, 103.52 s. Independent trace grounding audits are recorded
with the final gate. Coverage of unit controls
includes dynamic/shadowed receivers, bound argument slots, protected
budgets, cycles, determinism, inadequate anchors, true insufficiency,
metamorphic changes, counterfactual paths, timeouts and resource limits.

Completed external traces: evidence references 2/2 valid; unsupported
grounding claims 0/4; forbidden recommendations 0. Holdout had no claims or
references, so its separate grounding percentages are undefined, not 100%.
Across completed external decisions, filesystem mutations, tool dispatches,
authority grants, execution authorizations and destructive actions are all 0.
The 14-case synthetic regression independently reports all the same safety
counts as 0, evidence-reference validity 100%, unsupported claims 0% and
forbidden recommendations 0. Reference validity is not semantic correctness:
the completed external development recommendation was still wrong.

## GIT AND ARTIFACT PRESERVATION

Work was performed in `C:/Users/User/AppData/Local/Temp/JARVIS-v11` on main;
the older dirty `D:/JARVIS` checkout was not reset or synchronized. No
`_arquivo/`, credentials, temporary workspaces or large raw JSON are staged.

V12's 22 traces and portable ZIP remain verified outside Temp under
`C:/Users/User/AppData/Local/JARVIS/evaluation-artifacts/v12` (ZIP SHA-256
`4c4acc4f4f038283a7e388367402b7022b40474cffafff0c319308e6c49f4b05`).
V13's 32 traces are verified in the corresponding `v13` directory.
V14 raw gzip traces are content-addressed under the corresponding `v14`
directory; their final verified index is `v14-artifact-preservation.json`
(14 content-addressed compressed archives, including the aggregate synthetic
live trace). Reused retention/audit helper format version 13 is not a change
to the v14 corpus or runtime version.
All originals are retained. This is a verified same-host backup, not
off-machine disaster recovery. Historical reports/seals are not rewritten.

Commits:

- `5a9736e` — `test(predictive): add behavioral graph evaluation`
- `9c8f781` — `feat(predictive): resolve bounded api paths`
- Final stage-gate commit recorded in Git after reports and audits.

## FINAL GATES

Authoritative machine-readable checks: `v14-stage-gate.json`.
Coverage cannot pass without a nonempty independently known API-path
population, representable defect contracts, candidate recall, independent
noise adjudication, complete operational runs and grounding safety.
Decision readiness cannot pass on path/effect recall alone.

```text
PREDICTIVE_SYNTHETIC_READY = true
PREDICTIVE_EXTERNAL_COVERAGE_READY = false
PREDICTIVE_EXTERNAL_DECISION_READY = false
SIMULATION_RUNTIME_READY = false
```

Both external live populations are operationally incomplete; completed root
recall is 0/1 each, the development offline candidate recall remains 1/2,
and the held-out API-path denominator is empty. Independent contract/noise
adjudication is incomplete. These fail the preregistered coverage and
decision requirements. No Candidate Simulation was implemented or run.

The next missing abstraction is a verified behavioral contract/effect-to-
observation relation, including class API dispatch and mutation/iteration
consumption. Neither additional general ranking weights nor indiscriminate
retrieval expansion is justified by these results.
