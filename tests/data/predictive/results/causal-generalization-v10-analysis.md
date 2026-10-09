# Predictive Causal Generalization v10

## Generator architecture

Generator v4 produces tiny Python projects from typed `ProgramSpec`s.  The
`CausalOracle` is created before retrieval or reasoning and records root kind,
responsibility, defect-bearing relation, repair strategies, target roles,
argument slot, SCC membership, and evaluation population.  Seeds deterministically
randomize names, ordering, wrappers, comments, docstrings, and distractors.

The seven generated families are argument binding, return contract, the
argument/return factorial, import graph, producer/manifestation, repair target,
and negative controls.  Development has 172 structural cases; compositional
evaluation has 54 unseen factor combinations; the counterfactual set has 24
pairs; the complete holdout has 60 cases.

## V9 engine frozen results

Before engine hardening, deterministic generation already had 172/172 root
candidate recall, 102/102 argument-slot recall, and 10/10 SCC recall.  In the
42-case stratified live sample, however, the selector reached only 38/42 root,
repair-pair, and top1 validity.  Recommendation precision was 31/35.

| Family | Root validity |
| --- | ---: |
| argument binding | 4/6 |
| argument vs return | 4/6 |
| import graph | 6/6 |
| negative control | 6/6 |
| producer/manifestation | 6/6 |
| repair target | 6/6 |
| return contract | 6/6 |

This decomposed the problem as selection/identity, not retrieval or candidate
generation.

## Failure fingerprints and minimal counterexamples

1. `keyword/direct + temporary variable`: a concrete invalid actual argument
   lost its slot lineage and a satisfied producer return was treated as the
   defect.
2. `positional/wrapper-depth=2`: the first argument-binding edge was hidden by
   a later return edge, erasing the actual/formal slot identity.
3. `valid argument + invalid annotated return, direct`: a generic sibling
   binding survived against a demonstrated return-contract violation.
4. The same broken-return topology with one transparent wrapper reproduced the
   third failure, proving a family-level issue rather than a seed-specific one.

The reducer showed that neither distractors nor domain names were required for
these failures.

## Hardening

- Binding lineage now preserves the first `PASSED_AS_ARGUMENT` edge before
  traversing a return chain.
- Return-contract evidence is split into contract existence and deterministic
  violation; a satisfied annotation becomes manifestation-only.
- Concrete literal/alias return values are checked against simple annotations
  without inventing types for dynamic expressions.
- Semantically identical binding roots collapse by stable binding identity.
- Pairwise structural dominance prefers a proven contract violation over an
  unrelated sibling binding, while a satisfied return loses to the actual
  argument defect.
- Repair strategy compatibility is derived from the selected root and slice,
  not broad lexical compatibility.

No scoring weights changed.

## Layer decomposition after hardening

| Layer | Development | Compositional | Holdout combined |
| --- | ---: | ---: | ---: |
| retrieval / structural origin | 172/172 | 54/54 | 60/60 |
| root candidate generation | 172/172 | 54/54 | 60/60 |
| root selection | 42/42 | 54/54 | 60/60 |
| repair given correct root | 42/42 | 54/54 | 54/54 recommended |
| target given root + strategy | 42/42 | 54/54 | 54/54 recommended |
| end to end | 42/42 | 54/54 | 60/60 |

## Compositional and counterfactual generalization

All 54 unseen combinations passed root, repair-pair, and top1 validity.  The
seen/unseen gap is zero percentage points.  Across 24 argument-defect ↔
return-defect pairs, root, repair, and target sensitivity were each 24/24.
The paired metric requires both sides to be correct and semantically different,
so two unstable wrong answers cannot count as sensitivity.

## Holdout discipline

The original 48-case one-shot passed 48/48 but exposed a benchmark-design flaw:
its positional affordability cap left import and manifestation denominators at
zero.  No engine change followed.  The untouched seed tail `3048..3059` was
then frozen as a 12-case supplement and executed once.  It passed 12/12,
including 4/4 SCC precision and recall, 2/2 observer exclusion, and all four
negative import controls.  The combined holdout is 60/60 with SHA-256
`c706b90f35e8e0cc93762a42bf7ddbbd60bc0cb65036eca0bdab71145fec4706`.

## Regressions and safety

Evidence-reference validity remains 100%; unsupported-claim rate remains 0%;
forbidden recommendations remain zero.  Historical v7 recovery remains at
100% precision, 100% success, and 0% noise, with regression tests unchanged.
Filesystem mutations, tool dispatches, execution authorization, authority
grants, and destructive actions are all zero.

## Latency

Quality passed, but the stricter 45-second latency objective did not:
development averaged 46.6 s/Qwen request, compositional 64.7 s, and combined
holdout 59.1 s.  Each case still uses at most one Qwen call.  This is an
operational optimization backlog item, not a causal-quality failure.

## Stage gate

All predefined causal-quality, compositional, counterfactual, grounding, and
safety gates passed.  Predictive Consultative v1 is therefore frozen.  Runtime
simulation remains unavailable and fail-closed; the next milestone is Secure
Sandbox → Candidate Simulation → Observed Outcomes.

`PREDICTIVE_QUALITY_READY = true`

`SIMULATION_RUNTIME_READY = false`
