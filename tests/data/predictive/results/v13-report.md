# Symbol Coverage & Causal Candidate Generation v13

## Artifact preservation

All 22 original v12 traces and their portable ZIP have a verified copy at
`C:/Users/User/AppData/Local/JARVIS/evaluation-artifacts/v12`, outside Temp.
ZIP SHA256: `4c4acc4f4f038283a7e388367402b7022b40474cffafff0c319308e6c49f4b05`.
Original temporary files were retained. Final v13 preservation verifies 32
content-addressed traces at the sibling `v13` directory; twelve additional
historical-diagnostic traces are preserved under `v13/historical-diagnostic`.
The manifests record raw and compressed hashes. These are same-host backups,
not off-machine disaster recovery. Large traces are excluded from ordinary Git
reports; historical reports, source seals and reproducible definitions remain.

## Layer coverage before

On the twelve observed v12 defects, before any engine change:
index 12/12 files and symbols -> file retrieved 11/12 -> symbol selected 3/12
-> symbol/facts in ledger 3/12 -> correctly localized candidate 1/12.
First losses: eight symbol windows, one file retrieval, two candidate
generation failures, one observation without a detected loss. The local AST
edge proxy was 12/12; it plainly did not measure defect-relation correctness.

ProjectCandidateGenerator selects one slice per file; a retrieved file does
not transport every function. Context range intersection missed enclosing
functions. The ledger collapsed multiple windows to the last window and lost
qualified method ownership. Whole-file causal parsing could still produce
nodes without their relevant ledger facts. Behavioral APIs, mutations, yield,
exception handlers and dynamic method dispatch remain incompletely modeled.

## Symbol recovery architecture

RelevantSymbolContext records qualified identity, range, actual AST reads,
calls, returns, assignments, provenance and annotation presence. Exact reported
APIs/values anchor selection within retrieved files; uniquely resolved local
or imported calls can add bounded neighbors. Dynamic receivers and shadowed
imports are not guessed. Limits: one attempt, two extra files, sixteen symbols,
depth two, 240 lines/symbol and 48 KB additional context. The pre-existing
initial file/window selection is not silently claimed to obey this budget.

No scoring, policy, repair logic, analyzer, v7 recovery budget or v12 relevance
gate changed; normalized source hashes confirm those freezes. A first
implementation displaced an existing synthetic causal anchor and failed a
parameter-owner regression. The final implementation only supplies a fallback
anchor when the original pipeline has none. Both intermediate observations and
final observations are retained, not substituted for each other.

## Ledger enrichment

Disjoint windows are unioned per file. Qualified class/function ownership is
preserved. Existing call, return, assignment and condition facts are reused;
there is no invented AFFECTS_SYMPTOM evidence atom or defect assertion.
Lookup indexes preserve IDs, atom order and existing fallback semantics.
SYMBOL_RETRIEVED_NOT_PROVEN_DEFECTIVE is explicit. The additive JSON field
symbol_context_schema_version=1 does not remove existing schema-6 fields.

## Behavioral causal slicing

A unique reported API can seed existing return/condition nodes when no older
anchor exists. Its bridge is inferred, not deterministic proof of a violated
contract. No EXCEPTION_SITE is fabricated for behavioral reports. This is
limited: an existing but unsuitable generic anchor can still prevent the
fallback from using a better API. Real constructor/mutation/exception-helper
semantics cannot be inferred merely from having the correct definition.

## Oracle injection study

Final-runtime development interventions, isolated from real accuracy:

| Mode | Correct symbol in ledger | Correctly localized candidate |
|---|---:|---:|
| A normal initial pipeline | 2/6 | 0/6 |
| B adjudicated symbol/window | 6/6 | 1/6 |
| C actual AST facts only | 6/6 | 0/6 |

Mode D could supply grounded localized candidates in only 1/6 cases. That
single normal selector call chose the adjudicated location (1/1); the returned
strategy and target also match this case's location/strategy oracle. Five
cases are ORACLE_ROOT_NOT_REPRESENTABLE, not wrong selector decisions. D does
not fabricate absent causal paths. These are source-location proxies, not a
proof that the engine discovered the semantic defect. The earlier pre-fix
offline B result was 3/6; preserving the validated old anchor reduced it to
1/6. That tradeoff is disclosed, not hidden or used to inflate final metrics.

## Development before/after

Ten new independent defects, three repositories (pytest, Sphinx, Flask), one
original upstream formulation each: six development and four sealed holdout.
No corrective diff enters a runtime prompt. Python-only pre-fix snapshots are
hashed and parsed read-only; upstream fixes/tests adjudicate the oracle but
are not executed locally. The population is purposive, not representative.

| Metric | Frozen before | Final development | One-shot holdout |
|---|---:|---:|---:|
| Evaluable / total independent defects | 5/6 | 5/6 | 4/4 |
| Relevant file | 4/5 | 4/5 | 2/4 |
| Correct symbol in ledger | 2/5 | 2/5 | 0/4 |
| Correctly localized candidate | 0/5 | 0/5 | 0/4 |
| Selected adjudicated root | 0/5 | 0/5 | 0/4 |
| Repair-pair validity | 0/5 | 0/5 | 0/4 |
| Recommendation correctness | 0/2 | 0/2 | undefined (0/0) |
| Recommendation coverage | 2/5 | 2/5 | 0/4 |
| False abstention, positive cases | 3/5 | 3/5 | 4/4 |

The frozen Sphinx-7738 run exceeded fifteen minutes and did not finish; the
final run exceeded the preregistered 240-second whole-case operational budget.
Neither is counted as a root error or abstention. Completed raw observations
were recovered rather than fabricated. Before timings are unavailable; no
latency delta is invented. Conditional root selection is undefined in both
normal populations (no correct candidate), not a 0%-accurate selector.

Historical v12 diagnostic rerun: symbol/ledger coverage rises 3/12 -> 6/12
(+25 percentage points); candidate recall stays 1/12. This is transport progress,
not unseen generalization or decision progress. File-to-symbol conversion
rises 3/11 -> 6/11; ledger-to-candidate conversion is 1/6 after enrichment.

Noise is not semantically adjudicated. Explicit symbol contexts total 34 in
five evaluated development formulations and 64 in four holdout formulations;
all holdout contexts miss the oracle symbol. Off-oracle neighbors are not
automatically irrelevant, but budget saturation is a clear limitation.
Coverage fields describe the enriched initial context before v7 recovery;
`initial_files` is therefore not a pristine lexical-top-k measurement. Final
candidate availability is separately recorded. Exact pre-enrichment symbol
noise and oracle defect-edge recall are not established by these proxies.

## External holdout

Corpus SHA256: `8579df1e5991831dd30607deddab615658f11246c27b8d0a192209f1eff2af21`.
Corpus froze 2026-10-10 19:03:09 UTC, before engine changes; runtime/evaluator
froze 19:46:53 UTC. Pinned upstream commits, source hashes, adjudications and
PR references are in external_v13/cases.json. Exactly one holdout live-mode
execution occurred, with zero Qwen requests because no viable roots survived.
No post-holdout engine tuning occurred; the freeze hashes still match.

First losses: two FILE_NOT_RETRIEVED and two SYMBOL_WINDOW_MISSED. All four
also lack the correct candidate and falsely abstain. This is upstream of
ranking. Root-validity Wilson 95% intervals are [0, 43.4%] for 0/5 development
and [0, 49.0%] for 0/4 holdout; these intervals do not establish population
readiness. The observed coverage/candidate thresholds fail decisively.

## Synthetic regression

Final structural run: 172/172 root candidates, 102/102 argument identities,
10/10 SCCs. The generated suite includes repeated naming/surface draws.
Stratified live regression: root, repair-pair and top1 14/14; recommendations
12/12; two correct abstentions; argument identity 5/5; import precision/recall
2/2. Existing metamorphic, counterfactual, recovery and safety unit regressions
pass. The historical exhaustive live v10 results were not rewritten or rerun;
this is a limited regression, not a new exhaustive external qualification.

## Latency

Final development service wall mean 46.7 s (five completed cases); Qwen mean
51.0 s over two calls. Holdout service wall mean 22.9 s, zero model calls.
Mean symbol selection: 0.910 s development / 0.822 s holdout; context inclusive
0.990 / 0.850 s; ledger 8.635 / 4.386 s; causal construction 4.128 / 3.270 s.
Phases are inclusive and can overlap. Worker budgets include preflight
diagnostics whereas service-wall observations do not. Timing includes cProfile
and colocated evaluation load, not a controlled isolated interactive benchmark.
Synthetic Qwen mean 48.5 s over twelve requests. Latency and the large initial
basename/test context remain operational backlog; no causal weights were
changed to conceal the costs. Normal analysis uses at most one model call;
the one oracle call belongs to the separate diagnostic experiment.

## Security and tests

External final snapshots: filesystem mutations=0, tool dispatches=0,
execution_authorized=0, authority grants=0, destructive actions=0.
Forbidden recommendations=0. Independently audited output references 4/4
valid, unsupported grounding claims 0/6. Holdout emits no claims or references;
its empty rates are undefined, not falsely reported as perfect. Causal
correctness is distinct from grounding of canonical factual claims.

No third-party imports, installers, tests, patches or simulation executed on
the host. Docker executable is unavailable; sandbox fails closed. Existing
v11 foundation and production working trees were not modified.

Predictive unit suite: 306 passed, 2 skipped. Full suite: 1547 passed, 3 skipped,
one existing VOICE_TTS_SPEED deprecation warning. git diff --check passes.

## Conclusion

The bounded transport implementation is tested and improves historical symbol
coverage, but does not meet new external coverage or decision gates. Remaining
work is public-API-to-private-helper resolution, behavioral anchoring under
multiple plausible entry points, actual fact-to-node association, method/return
semantics, and controlled initial context cost. Neither more ranking points nor
oracle-supplied names belong in production logic.

PREDICTIVE_SYNTHETIC_READY = true (currently validated synthetic scope)

PREDICTIVE_EXTERNAL_COVERAGE_READY = false

PREDICTIVE_EXTERNAL_DECISION_READY = false

SIMULATION_RUNTIME_READY = false
