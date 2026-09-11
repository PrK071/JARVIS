# Predictive Causal Reasoning v3 — failure analysis

## Inputs and holdout discipline

- Pre-v3 baseline: `grounding-hardening-development-after.json` (30 cases).
- Causal development: 42 cases, including 12 new causal-flow cases.
- Historical holdouts v1/v2 were not used as unseen data.
- Holdout v3 contained 12 new cases and was sealed at SHA-256
  `ba6d5eab86a6c1277a490e6e5e1ca4d64990fa24f66abfeae560d7a5573aff6`
  before tuning. It was run live once, after development was frozen. No tuning
  followed its execution.

## Pre-change failure classes

Case review of the grounding-hardening report showed that retrieval was rarely
the bottleneck. The dominant classes were:

1. **Wrong causal direction** — the reasoner selected the return/expression on
   the traceback line rather than a producer, argument binding, configuration
   source, or branch that fed it.
2. **Missing data flow** — calls did not connect arguments to parameters,
   callee returns to caller assignments, method receivers to calls, or imported
   configuration names to their module assignment.
3. **Failure site treated as root** — a traceback proved where evaluation
   stopped but was scored as if it proved where the defective value originated.
4. **Repair family drift** — a grounded hypothesis could still produce a
   generic or incompatible strategy; validation often won because it was cheap.
5. **Lexical evaluation attraction** — several `WRONG_ROOT_CAUSE` and
   `WRONG_SOLUTION_FAMILY` labels represented vocabulary mismatch rather than a
   structurally wrong selected origin.
6. **No import/raise representation** — import cycles and guarded `raise`
   statements could end with no causal root and false abstention.

## Development hardening iterations

The first causal live iteration produced: hypothesis hit 52.4%, causal root hit
63.6%, repair strategy hit 90.9%, solution family hit 33.3%, ranking hit 29.0%,
recommendation precision 29.0%, coverage 81.6%, unsupported claims 0%, and
average Qwen latency 31.0 seconds.

General changes derived from those failures:

- upstream source distance and origin separation replaced traceback proximity
  as positive causal signals;
- failure-line return/attribute manifestations and unrelated test inputs were
  demoted;
- method receiver, nested call, positional/keyword binding, return, default,
  configuration, import and guarded-raise edges were added;
- repair compatibility, causal target validation and a root-specific repair
  strategy score were introduced before ranking;
- producer/return and consumer/boundary paraphrase families received stable
  structural signatures.

## Final development failures

Final metrics improved to causal root 81.8%, repair strategy 90.9%, solution
family 81.0%, ranking/recommendation precision 55.3%, with 100% coverage.
Remaining failure counts:

- `WRONG_RANKING` (17): chiefly `CORRECT_ARGUMENT` versus boundary-validation
  preferences in legacy cases; some are genuine strategy preference errors,
  while others expose incomplete structured strategy truth in the v1 corpus.
- `WRONG_ROOT_CAUSE` (13): a mix of lexical signal misses and unresolved
  semantic choices among multiple valid nodes on the same path.
- `WRONG_SOLUTION_FAMILY` (4): repair taxonomy/legacy lexical-family mismatch.
- `ROOT_CAUSE_SELECTION_ERROR` (1), `REPAIR_STRATEGY_ERROR` (1), and
  `REPAIR_TARGET_ERROR` (1): `PC3D-004`, where the problem describes a default
  binding while the fixture also contains a structurally complete explicit
  upstream caller path.
- `CAUSAL_SLICE_MISSED_ORIGIN` (1): `PC3D-011`; the expected control-flow origin
  is not structurally connected to the unconditional return at the reported
  failure line.

These classes block simulation because development hypothesis hit (59.5%),
ranking hit (55.3%), and recommendation precision (55.3%) remain below their
75%/65%/65% gates.

## Holdout v3 (single live run)

The sealed holdout passed its configured quality gate: hypothesis 83.3%, causal
root 81.8%, solution family 83.3%, repair strategy 90.9%, ranking and
recommendation precision 100%, coverage 90.9%. Remaining failures were one
retrieval miss (`PC3H-003`), one wrong root (`PC3H-009`), and two root-selection
failures (`PC3H-010`, `PC3H-011`), with the latter producing one false
abstention. These results were recorded without post-holdout tuning.

## Safety and grounding

Both final live runs had 100% evidence-reference validity, 0% unsupported
claims, 100% candidate-family diversity, zero forbidden recommendations, zero
filesystem mutations, zero tool dispatches, zero authority grants, zero
execution authorization, and zero destructive actions.
