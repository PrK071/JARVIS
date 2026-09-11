# Predictive benchmark v4

Version 4 is an adjudication layer over the immutable v3 cases plus a newly
sealed holdout. Ground truth is derived from fixture source, tests, traceback,
and demonstrable causal flow. Model output is never an adjudication source.

Every development case has one record in `adjudications/development.jsonl`.
The record separates an acceptable diagnosis/repair from a preferred one and
pairs each repair strategy with the files and symbols it may validly target.

Statuses have evaluation semantics:

- `CANONICAL`: one technically established diagnosis exists.
- `MULTIPLE_VALID_ANSWERS`: more than one diagnosis or repair is defensible.
- `AMBIGUOUS`: the fixture cannot distinguish competing conclusions.
- `INCOMPLETE`: the previous truth lacks the observations required to score a
  causal origin.
- `INVALID`: the case does not measure the failure it claims to measure.

Only canonical and multiple-valid records contribute to v4 quality gates.
Ambiguous, incomplete, and invalid records remain visible as benchmark defects;
they are never silently counted as engine errors.

## Metrics

- `root_cause_validity`: a selected structured root matches any acceptable
  origin path, optional symbol, and cause kind.
- `root_cause_preference_hit`: a selected root matches a preferred origin when
  the fixture justifies a preference.
- `repair_validity`: at least one candidate matches a strategy-target pair.
- `repair_preference_hit`: at least one candidate matches a preferred pair.
- `top1_is_valid` / `top1_is_preferred`: correctness and preference of the
  recommendation are measured separately.
- `ranking_preference_hit`: top-1 is preferred, or an explicit ambiguity is
  accepted where multiple valid repairs have no justified preference.
- `recommendation_validity_precision`: valid recommendations divided by all
  recommendations on evaluable positive cases.
- `recommendation_coverage`: recommendations divided by evaluable positive
  cases.
- `hypothesis_hit_v1_lexical` remains diagnostic only and is not a v4 gate.

The v4 holdout contains 18 new cases. Its SHA-256 includes the case JSONL, its
v4 adjudication JSONL, and every referenced fixture file. It must be evaluated
live once only after any development-set selector work is frozen.
