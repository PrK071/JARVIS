# Structural Recovery v7 stage gate

## Scope and freeze

- Engine base: `8c6d1382a1b32f0c7f585c260213800d8cc2c620`
- Development cases: 72 total, 54 positive/evaluable
- Holdout v7: 20 total, 16 positive/evaluable
- Holdout v7 SHA-256: `6f80663ca668b9d52451c1430a6c1d0fe8c32a4d34967db74fbda6dcdcf99839`
- Holdout frozen at: `2026-09-12T15:20:00-03:00`
- Holdout live executions in this phase: 1
- Tuning after holdout execution: none

## Failure funnel before

The six positive false abstentions from the historical v6 holdout failed before
the reasoner:

| Loss stage | Cases |
|---|---:|
| `INITIAL_RETRIEVAL_INCOMPLETE` | 2 |
| `CAUSAL_SLICE_INCOMPLETE` | 4 |
| Root selection or later | 0 |

Root causes were two missing unique-symbol seeds, two unique-basename traceback
paths that were not normalized, and two real production import SCCs whose
creation incorrectly depended on problem wording.

## Structural recovery architecture

`StructuralRecoveryService` receives the initial `ProblemContext`, existing
project snapshot, Evidence Ledger and Causal Slice. It creates typed unresolved
frontiers and expands only verified callers, callees/return producers,
arguments, attribute origins, import neighbors, config origins or unique symbol
references. Expansion is deterministic and reuses the existing index.

Budgets are fail-closed: one attempt, depth 2, four extra files, twelve symbols
and thirty EvidenceAtoms. The recovered context is preserved beside the initial
context in `DecisionReport.recovery`; no tool, writer, subprocess or execution
authority is available to this service.

The causal path budget was raised from eight to twelve nodes so two trivial
forwarding wrappers do not hide a proven producer. A unique complete upstream
return producer may dominate an intermediate forwarding return, while parallel
producers remain ambiguous.

## Development v7

| Metric | Before hardening | After |
|---|---:|---:|
| Root-cause validity | 79.6% | 92.6% (50/54) |
| Repair-pair validity | 81.5% | 94.4% (51/54) |
| Top-1 validity | 77.8% | 90.7% (49/54) |
| Recommendation validity precision | 77.8% | 90.7% (49/54) |
| Recommendation coverage | 100.0% | 100.0% (54/54) |
| False abstention | 0.0% | 0.0% (0/54) |
| Evidence reference validity | 100.0% | 100.0% |
| Unsupported claim rate | 0.0% | 0.0% |

The “before” column is the first v7 live run after adding recovery but before
the bounded path/dominance correction. It is not a v6-compatible metric.

### Recovery

| Metric | Result |
|---|---:|
| Attempt rate | 20.4% (11/54) |
| Success rate | 100.0% (11/11) |
| Precision of changed decisions | 100.0% (11/11) |
| Noise rate | 0.0% |
| Average extra files | 1.36 |
| Average extra symbols | 1.45 |
| Average extra EvidenceAtoms | 4.27 |
| False abstention before recovery | 14.8% (8/54) |
| False abstention after recovery | 0.0% (0/54) |

Recovery actions across positive cases: `EXPAND_CALLEE=6`,
`EXPAND_ATTRIBUTE_ORIGIN=3`, `EXPAND_IMPORT_NEIGHBOR=5`, and
`EXPAND_CALLER=1`. A case can use more than one frontier.

The evaluation definition is strict: in live mode, recovery succeeds only when
the final root cause and final recommendation are structurally valid. Merely
adding files or producing candidates is not success.

### Precision/coverage frontier

| Policy | Precision | Coverage | False abstention |
|---|---:|---:|---:|
| No recovery (counterfactual from initial sufficiency) | 89.1% (41/46) | 85.2% (46/54) | 14.8% |
| Strict bounded recovery (shipped) | 90.7% (49/54) | 100.0% (54/54) | 0.0% |
| Moderate/unbounded expansion | Not enabled | Not enabled | Not enabled |

No moderate mode was invented because the strict policy already recovered all
development abstentions without noise.

## Metamorphic and counterfactual development

The existing versioned robustness suite was rerun live against the v7 engine:
70 metamorphic variants and 10 counterfactual pairs.

| Metric | Result |
|---|---:|
| Root-cause invariance | 91.4% (64/70) |
| Repair invariance | 94.3% (66/70) |
| Target invariance | 94.3% (66/70) |
| Problem-wording invariance | 83.3% (10/12) |
| Counterfactual root sensitivity | 90.0% (9/10) |
| Counterfactual repair sensitivity | 90.0% (9/10) |
| Counterfactual target sensitivity | 90.0% (9/10) |
| Variant root validity | 83.6% (51/61) |
| Variant repair-pair validity | 86.9% (53/61) |
| Variant top-1 validity | 80.3% (49/61) |
| Variant recommendation precision | 80.3% (49/61) |
| Variant false abstention | 0.0% (0/61) |

Candidate/evidence/identifier ordering remains covered by deterministic unit
regressions; the full robustness run did not add redundant order-bias Qwen
calls. Remaining metamorphic blockers are docstring/error wording changes and
the single wrapper-path transformation, not candidate list ordering.

## Holdout v7 one-shot

| Metric | Result | Gate |
|---|---:|---:|
| Root-cause validity | 75.0% (12/16) | >= 80%: fail |
| Repair-pair validity | 75.0% (12/16) | >= 80%: fail |
| Top-1 validity | 80.0% (12/15 recommendations) | >= 80%: pass |
| Recommendation validity precision | 80.0% (12/15) | >= 80%: pass |
| Recommendation coverage | 93.8% (15/16) | informational |
| False abstention | 6.2% (1/16) | <= 10%: pass |
| Evidence reference validity | 100.0% | hard pass |
| Unsupported claim rate | 0.0% | hard pass |

Recovery attempted four positive cases and produced four valid final decisions:
success 100%, precision 100%, noise 0%, average 1.5 extra files, 1.5 symbols
and 4.5 atoms. Initial structural sufficiency covered 75%; strict recovery
raised recommendation coverage to 93.8%.

The final funnel contains 15 recommendations, four correct true-insufficient
abstentions, and one `INITIAL_RETRIEVAL_INCOMPLETE` false abstention
(`PR7H-016`). Remaining failures are `ROOT_CAUSE_SELECTION_ERROR=3`,
`REPAIR_TARGET_ERROR=3`, `RANKING_ERROR=3`, and `FALSE_ABSTENTION=1`.
These are recorded as next-iteration backlog; no holdout-specific tuning was
performed.

## Latency and tokens

| Split | Normal case latency | Recovery case latency | Qwen average | Prompt tokens/request |
|---|---:|---:|---:|---:|
| Development | 27.27 s | 29.50 s | 27.90 s | 1,447 |
| Holdout v7 | 30.35 s | 24.08 s | 29.97 s | 1,692 |

Recovered prompts averaged 1,720 tokens in development and 1,734 on holdout;
normal prompts averaged 1,374 and 1,677 respectively. Recovery still uses one
Qwen call per eligible case.

## Safety

Across development, robustness and holdout artifacts:

```text
evidence_reference_validity = 100%
unsupported_claim_rate = 0%
forbidden_candidate_recommendations = 0
filesystem_mutations = 0
tool_dispatches = 0
execution_authorized = 0
authority_grants = 0
destructive_actions = 0
```

## Gate

Development canonical and recovery gates pass. Development metamorphic quality
and holdout root/repair validity do not. Candidate Simulation remains out of
scope and the sandbox provider remains fail-closed/unavailable.

```text
PREDICTIVE_QUALITY_READY = false
SIMULATION_RUNTIME_READY = false
```
