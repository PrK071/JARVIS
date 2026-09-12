# Predictive robustness corpus v6

This corpus measures structural invariance separately from causal sensitivity.
Development contains 22 base cases, 70 controlled metamorphic variants, and 10
counterfactual pairs. `holdout_v6` contains 20 sealed metamorphic variants from
10 historical semantic bases and five counterfactual pairs. The transformations
and structural expectations are sealed together with the base cases, v5
adjudications, and fixture bytes. The holdout is evaluated live once only.

Metamorphic results compare causal roles (`RootCauseKind`, origin role,
`RepairStrategyKind`, `RepairTargetKind`, and causal-path node kinds), never
free-form text. Counterfactual pairs require decisions to change when the
fixture's selected causal defect changes.

Source fixtures remain immutable. Variants are materialized in temporary
directories, evaluated read-only, and deleted after each run.

Reports keep canonical quality, metamorphic invariance, counterfactual
sensitivity, exact/decision stability, ordering probes, safety counters, and an
abstention coverage funnel separate. A percentage is never emitted without its
eligible population in canonical benchmark v6 reports.
