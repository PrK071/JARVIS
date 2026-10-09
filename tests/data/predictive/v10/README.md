# Predictive causal generalization v10

This benchmark generates tiny Python projects from causal specifications.  The
expected answer is constructed with the program, before Project Intelligence or
the reasoner sees it.  No expected value is inferred from model prose.

## Populations

- `development`: 172 cases across argument binding, return contracts, the
  argument/return factorial, import graphs, producer/manifestation chains,
  repair targets, and negative controls.
- `compositional`: combinations that are absent from development, including
  mixed arguments with deeper wrappers and larger import SCCs.
- `holdout_v10`: 48 sealed specifications.  Its hash covers generator version,
  seeds, and all factor definitions.  Live evaluation is one-shot.
- `holdout_v10_supplement`: the 12 untouched seeds immediately after the
  original positional cap.  They complete the previously zero denominators
  for import SCC, manifestation, and negative controls without changing the
  engine between runs.
- counterfactuals: 24 argument-defect/return-defect pairs with the same surface
  family and an intentionally changed defect-bearing relation.

Generated projects live only in temporary directories.  The repository stores
the generator, seed/factor manifest, and result reports rather than hundreds of
expanded fixtures.

## Oracle

`CausalOracle` records root kind, responsibility, origin role, defect-bearing
relation, valid/preferred repairs, valid/preferred target roles, SCC membership,
and argument slot.  Single-root, multiple-root, and true-abstention populations
are reported separately.

## Layer metrics

| Metric | Numerator | Denominator |
| --- | --- | --- |
| `retrieval_accuracy` | expected origin represented in retrieved files | generated cases |
| `oracle_root_candidate_recall` | oracle root exists, or no root exists for a negative control | generated cases |
| `oracle_relation_recall` | matching root carries the oracle defect-bearing relation | generated cases |
| `oracle_slot_identity_recall` | matching binding preserves the defective ordinal/keyword | binding cases |
| `oracle_scc_recall` | matching root preserves exact production SCC members | runtime-cycle cases |
| `root_selection_given_oracle_candidate` | selected root matches oracle when a correct candidate existed | live evaluable cases |
| `repair_given_correct_root` | selected strategy is valid | cases whose selected root is correct |
| `target_given_correct_root` | target role is valid | cases whose selected root and strategy are correct |

Every reported rate carries `numerator`, `denominator`, and `population`.
Lexical similarity is not an oracle and no LLM judge participates in pass/fail.

## Commands

```text
jarvis predictive-generated-eval --mode structural --split development
jarvis predictive-generated-eval --mode live --split development --limit 42
jarvis predictive-generated-eval --mode structural --split compositional
jarvis predictive-generated-eval --mode live --split holdout_v10
jarvis predictive-generated-eval --mode live --split holdout_v10_supplement
```

Live mode is explicit and makes one structured Qwen call per case.  Structural
mode evaluates retrieval, ledger, slice, and root generation without Qwen.
