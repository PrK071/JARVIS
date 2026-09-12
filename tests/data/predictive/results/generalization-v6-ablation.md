# Predictive generalization v6 - ablation log

This log records development-only experiments. No holdout v6 result was read
while making these decisions.

## Metric denominator audit

The apparent v5 mismatch (`root_cause_validity=31.2%` versus
`top1_validity=72.7%`) was caused by different populations. Root validity used
all 16 positive cases, including five false abstentions. Top-1 validity used
only the 11 cases that emitted a recommendation. Benchmark v6 records the
numerator, denominator, eligibility rule, and population counts for every
canonical gate metric.

## Ablations

| Feature | State | Root validity | Repair-pair validity | Top-1 validity | Decision |
|---|---:|---:|---:|---:|---|
| Require an argument-related problem-text trigger before comparing binding roots | ON | 90.2% (37/41) | 90.2% (37/41) | 87.8% (36/41) | KEEP as weak prior pending stronger structural disambiguation |
| Require an argument-related problem-text trigger before comparing binding roots | OFF | 80.5% (33/41) | 85.4% (35/41) | 78.0% (32/41) | REJECT: comparison activated in unrelated flows |
| Boundary/validation words restrict the model schema to `VALIDATE_BOUNDARY` | ON | not promoted | not promoted | not promoted | RETAIN only as an explicit ablation switch |
| Boundary/validation words restrict the model schema to `VALIDATE_BOUNDARY` | OFF | 90.2% (37/41) | 90.2% (37/41) | 87.8% (36/41) | DEFAULT: user wording no longer eliminates structurally valid repairs |

The rejected unbounded-binding run is preserved in
`generalization-v6-ablation-unbounded-binding.json`. It demonstrates why a
new heuristic is not accepted solely because it fixes one wording variant.

The boundary rule had no structural proof: it encoded a repair answer from
user wording directly into the JSON schema. Its canonical score change is
small enough that architecture and metamorphic robustness take precedence.
The switch remains available only so future evaluations can reproduce the
ON/OFF comparison.

## Lexical provenance audit

| Signal | Classification | Constraint |
|---|---|---|
| Traceback path and line | `ESSENTIAL_INPUT` | Identifies a failure site, never the origin by itself |
| Import-error wording | `STRUCTURAL_CONFIRMATION_REQUIRED` | May start graph inspection; only an actual SCC proves a cycle |
| Boundary/validation wording | `UNSAFE_SHORTCUT` | Disabled by default; cannot restrict repair strategy |
| Symbol/contract words | `WEAK_PRIOR` | Cannot create evidence, roots, or repairs |
