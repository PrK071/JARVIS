# Predictive repair generalization v5 stage gate

## Protocol

The v5 holdout was sealed before engine hardening with SHA-256
`9045cf57ff0cf4139c937f0530922dfdf5755e56bb7b0f5846fc57f3dddbf3cd`
at `2026-09-11T19:49:43.5503554Z`. Development was evaluated and the engine
was frozen before the holdout was evaluated live once. No engine or corpus
changes were made after observing the holdout result.

## Development before and after

| Metric | Before | After | Gate |
|---|---:|---:|---:|
| Root-cause validity | 80.5% | **90.2%** | >= 85% |
| Repair strategy validity | 92.7% | **97.6%** | >= 90% |
| Repair target validity | 92.7% | **95.1%** | >= 85% |
| Repair pair validity | 85.4% | **92.7%** | >= 85% |
| Top-1 validity | 85.0% | **90.2%** | >= 85% |
| Recommendation validity precision | 85.0% | **90.2%** | >= 85% |
| Recommendation coverage | 97.6% | **100.0%** | >= 70% |
| False abstention | 2.4% | **0.0%** | <= 5% |
| Evidence reference validity | 100.0% | **100.0%** | 100% |
| Unsupported claim rate | 0.0% | **0.0%** | <= 5% |
| Import-cycle detection recall | 80.0% | **100.0%** | diagnostic |
| Average Qwen latency | 26.9 s | **27.2 s** | <= 45 s |

The development gate passed with no blockers. The material changes were:

- structurally equivalent root candidates are deduplicated before selection;
- import-cycle intent accepts traceback wording that describes an import
  triggering a cycle without requiring the exact phrase `import cycle`;
- a demonstrated upstream return contract or unique `None`/configuration
  origin can dominate a downstream manifestation;
- explicit parameter ownership can dominate unrelated flows in the same tiny
  repository;
- repair target selection distinguishes argument call sites from validation
  boundaries.

## Holdout v5 one-shot

| Metric | Result | Gate |
|---|---:|---:|
| Root-cause validity | **31.2%** | **>= 80%** |
| Repair strategy validity | **62.5%** | **>= 80%** |
| Repair target validity | **56.2%** | **>= 75%** |
| Repair pair validity | **56.2%** | **>= 75%** |
| Top-1 validity | **72.7%** | **>= 75%** |
| Recommendation validity precision | **72.7%** | **>= 75%** |
| Recommendation coverage | 68.8% | diagnostic |
| False abstention | **31.2%** | **<= 5%** |
| Evidence reference validity | 100.0% | 100% |
| Unsupported claim rate | 0.0% | <= 5% |
| Forbidden recommendations | 0 | 0 |
| Average Qwen latency | 28.1 s | <= 45 s |

The one-shot holdout failed the predictive quality gate. It was not rerun and
the engine was not tuned after this observation.

## Remaining failures

| Failure class | Cases | Responsible layer |
|---|---:|---|
| `FALSE_ABSTENTION` | 5 | structural sufficiency / candidate eligibility |
| `ROOT_CAUSE_SELECTION_ERROR` | 4 | root semantic identity and selector |
| `RANKING_ERROR` | 3 | repair pair preference / ranking |
| `CAUSAL_SLICE_ERROR` | 1 | argument-flow origin |
| `MISSED_STRUCTURAL_NEIGHBOR` | 1 | attribute-origin representation |
| `REPAIR_TARGET_ERROR` | 1 | repair target derivation |
| `FAILED_TO_ABSTAIN` | 1 | import-cycle negative case |

Concrete observations:

- `PC5H-002` selected the literal `None` origin instead of the expected return
  contract. The repair remained producer-local, but structural root identity
  did not match.
- `PC5H-003`, `PC5H-006`, `PC5H-010`, `PC5H-017`, and `PC5H-018` abstained even
  though their fixtures contain an evaluable causal origin.
- `PC5H-007` treated a function-local import as a runtime cycle in a negative
  case and should have abstained.
- `PC5H-011` found the correct file and `None` flow but did not preserve the
  expected attribute-level origin identity.
- `PC5H-014` through `PC5H-016` found the correct configuration file, but the
  selected configuration root omitted the `RETRY_DELAY` symbol required by
  the structural truth.

These failures are recorded as backlog for a future development corpus. They
must not be used to tune against the now-observed holdout v5.

## Safety and sandbox

Both live reports recorded zero filesystem mutations, tool dispatches,
execution authorizations, authority grants, destructive actions, and forbidden
candidate recommendations. Evidence reference validity remained 100% and the
unsupported claim rate remained 0%.

The sandbox capability check failed closed because Docker is unavailable. No
write confinement, network denial, environment sanitization, child-process
containment, resource limits, or cleanup capability was claimed as verified.

## Decision

`QUALITY_READY = false`

`SANDBOX_READY = false`

`NOT READY FOR CANDIDATE SIMULATION`

Candidate Simulation was not executed. The quality gate failed independently
of the secure-runtime gate, and the unavailable sandbox provider correctly
prevented host execution fallback.
