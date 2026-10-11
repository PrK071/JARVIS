# v15 contract audit before hardening

Baseline: `76a15a4c8aeba3385a8d001903037defd0cbf99f`.
No engine changes precede this audit. Historical v12-v14 results are retained.

| Layer | Observed representation | Missing distinction |
|---|---|---|
| Symptom | report, exception/behavior kind, optional source location | expected behavior is always unset |
| API graph | exact definitions, resolved calls, AST effects | reachability does not establish execution or responsibility |
| Ledger | definitions, calls, returns, assignments, conditions, effects | no typed expectation/application/violation record |
| Causal slice | disconnected callable/effect subgraph | no justified effect-to-observation bridge for state/yield APIs |
| Roots | backward paths to exception or return/condition anchor | CALLABLE and BEHAVIORAL_EFFECT explicitly excluded as roots |
| Service | no roots ends before Qwen | cannot express grounded investigation without a diagnosis |
| Operations | elapsed timings only after functions return | terminated workers lose the current phase and memory context |

Effects cannot generate roots merely by existing: that would recreate the v12
SCC error. A user expectation establishes an observation, not a violated source
contract. Primitive return annotations plus an unconditional literal return can
support a narrow static incompatibility. Conditional execution, dynamic state,
third-party semantics and arbitrary assertions generally need additional evidence.
The existing return-contract heuristic and validated synthetic path remain frozen;
the new representation must not silently reinterpret those historical decisions.

The safe intervention is an additive investigation-only branch when verified API
definitions, effects and implementation paths exist but the defect bridge does
not. Its JSON response selects supplied IDs and missing-evidence codes, never a
free-form diagnosis, root, score, patch or recommendation. No anchored evidence
means no model call. Promotion must validate actual represented root relations;
model confidence is never evidence. General scoring/recovery/policy are unchanged.

Operational limitation: v14 censored two workers at 240 s, but its historical
artifacts cannot distinguish their last deterministic phase from model queue or
generation. Do not fabricate that attribution. Persist start/end/error events
outside external snapshots; queue/generation remain unknown unless measured.

New external corpus must be independently sealed before runtime hardening. A
small controlled-family gain is not external diagnostic readiness. Full external
completion and independent hypothesis-location adjudication remain prerequisites
for investigation readiness, not optional labels to infer from coverage alone.
