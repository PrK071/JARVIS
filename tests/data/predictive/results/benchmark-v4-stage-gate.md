# Predictive benchmark v4 stage gate

## Protocol

The v4 holdout was sealed before selector tuning with SHA-256
`adbb18e043e6977403afd41c8b21a48bc254cc1fb83697fd8c2310557e49bf3f`.
It was evaluated live once, against selector commit `10e6b19`, after development
was frozen. No engine or holdout changes were made after observing the result.

## Development

| Metric | Result |
|---|---:|
| Root-cause validity | 100.0% |
| Repair validity | 100.0% |
| Top-1 validity | 100.0% |
| Ranking preference hit | 100.0% |
| Recommendation validity precision | 100.0% |
| Recommendation coverage | 100.0% |
| Evidence reference validity | 100.0% |
| Unsupported claim rate | 0.0% |
| Average Qwen latency | 27.3 s |

## Holdout v4 one-shot

| Metric | Result | Gate |
|---|---:|---:|
| Root-cause validity | 75.0% | >= 75% |
| Repair validity | **68.8%** | **>= 75%** |
| Top-1 validity | 78.6% | >= 70% |
| Ranking preference hit | 68.8% | >= 55% |
| Recommendation validity precision | 78.6% | >= 70% |
| Recommendation coverage | 87.5% | diagnostic |
| Evidence reference validity | 100.0% | 100% |
| Unsupported claim rate | 0.0% | <= 5% |
| Average Qwen latency | 26.4 s | <= 60 s |

The holdout failed only the repair-validity quality gate. All hard grounding and
safety gates passed.

## Remaining engine failures

| Failure class | Cases | Responsible layer |
|---|---:|---|
| `FALSE_ABSTENTION` | 2 | candidate target validation / retrieval |
| `ROOT_CAUSE_SELECTION_ERROR` | 2 | causal selector |
| `CAUSAL_SLICE_ERROR` | 1 | import-flow causal slice |
| `REPAIR_TARGET_ERROR` | 2 | repair target representation |

Concrete cases:

- `PC4H-008`: the control-flow root was valid, but target validation rejected
  all generated repairs.
- `PC4H-009` and `PC4H-012`: the reasoner chose a plausible value/`None` node
  rather than the benchmark's structural return-contract origin.
- `PC4H-011`: the slice attributed the circular import to the test import
  manifestation rather than either production import edge.
- `PC4H-014`: retrieval missed the relevant file and the engine abstained.
- `PC4H-017` and `PC4H-018`: valid repair mechanisms targeted the parameter
  `values`, while the structural repair truth names the containing function;
  target granularity needs an explicit representation in a future benchmark
  revision rather than post-hoc holdout relabeling.

## Decision

`NOT READY FOR CANDIDATE SIMULATION`

Candidate Simulation was not implemented because the predictive quality gate
failed. Independently, the repository has path allowlisting and delegated Codex
sandbox policies, but no local execution sandbox proving network denial,
environment sanitization, process containment, resource limits, and confined
writes for untrusted fixture tests.
