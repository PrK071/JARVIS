# Causal Identity & Responsibility v9 — Stage Gate

Holdout v9 was sealed before tuning and opened once after development was
frozen. Corpus hash:
`6c1ebb9bb5d652f27a0c20c2ef8ae27b33672250347d090c66f0075ac761d54c`
(24 cases, exact manifest match).

## Failure analysis before

The pre-change audit found overlapping failure classes. Import SCC failures
contributed three identity/generation cases; argument-vs-return contributed two
selection cases; sibling arguments contributed one identity/target case; one
upstream case contributed selection/abstention; and three ranking errors were
downstream consequences of an already-wrong root or target.

| Class | Cases | Finding |
|---|---:|---|
| Identity | 4 | Three module SCCs and one sibling argument slot |
| Generation | 3 | Import cycle represented as callable/edge, not SCC |
| Selection | 3 | Two argument-vs-return choices plus one upstream abstention |
| Target | 1 | Sibling slot evidence was transferred to the wrong parameter |
| Ranking | 3 | Downstream; no isolated scoring failure |
| Abstention | 1 | Producer path existed but final selection was empty |

## Import SCC model

Import cycles now use a module-level SCC identity with production edges,
observer edges and provenance. Test imports remain observers; `TYPE_CHECKING`
and local imports retain distinct provenance. A cycle root uses
`IMPORT_GRAPH_DEFECT` and targets an `IMPORT_EDGE`/module rather than a
callable from the traceback.

## Causal responsibility and argument slots

Root identity, causal role and causal responsibility are separate. The v9
profile records the defect-bearing relation and distinguishes
`ARGUMENT_SOURCE_DEFECT`, `ARGUMENT_BINDING_DEFECT`,
`RETURN_CONTRACT_DEFECT`, `IMPORT_GRAPH_DEFECT` and manifestation-only roots.
Argument bindings retain call site, callee, actual argument, formal parameter,
ordinal and keyword identity. Repair-target preference follows responsibility:
argument sources prefer call sites, external boundaries prefer parameters, and
default bindings prefer their exact formal parameter.

## Development v9

The v9 gate population contains 15 positive and 3 legitimate-abstention cases.

| Metric | Result |
|---|---:|
| Root validity | 15/15 (100%) |
| Repair strategy validity | 15/15 (100%) |
| Repair target validity | 15/15 (100%) |
| Repair pair validity | 15/15 (100%) |
| Top-1 validity | 15/15 (100%) |
| Recommendation precision | 15/15 (100%) |
| Coverage | 15/15 (100%) |
| False abstention | 0/15 (0%) |
| Import SCC identity | 5/5 (100%) |
| Argument vs return | 9/9 (100%) |
| Argument binding identity | 6/6 (100%) |
| Argument slot target | 6/6 (100%) |
| Root pairwise accuracy | 10/10 (100%) |
| Average Qwen latency | 18.36 s/request (16 requests) |

## Metamorphic and counterfactual v9

Development used 18 variants and four counterfactual pairs.

| Metric | Development | Holdout v9 |
|---|---:|---:|
| SCC invariance | 18/18 relevant checks aggregate to 100% | 100% |
| Root responsibility invariance | 18/18 (100%) | 17/18 (94.4%) |
| Argument-binding invariance | 100% | 100% |
| Repair invariance | 18/18 (100%) | 17/18 (94.4%) |
| Target invariance | 18/18 (100%) | 17/18 (94.4%) |
| Wording invariance | 6/6 (100%) | 6/6 (100%) |
| Argument/return switch sensitivity | 2/2 (100%) | 2/2 (100%) |
| SCC removal/addition sensitivity | 2/2 (100%) | 2/2 (100%) |

Absolute holdout variant validity remained weak: root `12/18` (66.7%), repair
pair `16/18` (88.9%), top-1 `16/18` (88.9%) and recommendation precision
`12/18` (66.7%). Stability therefore does not override the canonical quality
failure.

## Ablations

| Feature | ON root | OFF root | Direct v9 impact |
|---|---:|---:|---|
| Module/SCC identity | 100% | 66.7% | SCC identity 100% → 0% |
| Responsibility profile | 100% | 0% | argument-vs-return 100% → 0% |
| Argument slot identity | 100% | 80% | binding and slot accuracy 100% → 0% |

The features are retained because each contributes independent structural
signal and all metamorphic/counterfactual development checks remain at 100%.

## Holdout v9 — one shot

| Metric | Numerator / denominator | Result | Gate |
|---|---:|---:|---|
| Root validity | 12/20 | 60.0% | FAIL (≥85%) |
| Repair pair validity | 14/20 | 70.0% | FAIL (≥85%) |
| Top-1 validity | 14/20 | 70.0% | FAIL (≥85%) |
| Recommendation precision | 12/18 | 66.7% | FAIL (≥85%) |
| Coverage | 18/20 | 90.0% | diagnostic |
| False abstention | 2/20 | 10.0% | FAIL (≤5%) |
| Import SCC identity | 5/6 | 83.3% | FAIL (≥90%) |
| Argument vs return | 6/12 | 50.0% | FAIL (≥85%) |
| Argument binding identity | 5/8 | 62.5% | FAIL (≥90%) |
| Argument slot target | 5/8 | 62.5% | FAIL (≥90%) |
| Root pairwise accuracy | 7/13 | 53.8% | FAIL (≥80%) |
| Average Qwen latency | 19 requests | 42.57 s | PASS (≤45 s) |

Residual positive-case failures are concentrated in null-return origin,
keyword/sibling/default/mixed slots, one import observer case, and two wording
cases. There were two false abstentions and one negative no-cycle case that
failed to abstain. No post-holdout tuning was performed.

## Ranking and recovery regression

All four holdout `WRONG_RANKING` codes occurred after an incorrect root and/or
repair pair; there was no isolated case where a correct root/target pair lost
solely because of the frozen scoring formula. General scoring weights were not
changed.

Structural Recovery v7 remains frozen. Its recorded development regression
gate remains: 11/11 successful recoveries, 100% precision, 100% success and 0%
noise. Predictive tests covering recovery remain green.

## Grounding and safety

Canonical development and holdout both retained 100% evidence-reference
validity and 0% unsupported claims. Across canonical and robustness runs:

```text
filesystem_mutations = 0
tool_dispatches = 0
execution_authorized = 0
authority_grants = 0
destructive_actions = 0
forbidden_candidate_recommendations = 0
```

## Tests and decision

Predictive suite: `180 passed`.

Full suite: `1422 passed, 1 skipped, 1 warning`.

Development and robustness gates pass, but the sealed canonical holdout fails
root responsibility, exact argument-slot targeting, repair pair, top-1,
recommendation precision and false-abstention gates.

```text
PREDICTIVE_QUALITY_READY = false
SIMULATION_RUNTIME_READY = false
```

Candidate Simulation remains out of scope. The next predictive iteration must
address the holdout failure classes through a newly sealed development cycle;
secure sandbox work remains a separate prerequisite after quality readiness.
