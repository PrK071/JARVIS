# Semantic Selection & Target Robustness v8 — Stage Gate

Date: 2026-09-18

Engine evaluated: `1daabe6036dbcc672d7fcf3ccd2e2ed9f9c1c7be`

Holdout v8 SHA-256: `8e3635691f5bc112a948e2ac0a0fb5829e6a5587a0e67945066c832be35867a3`

## Protocol

- `holdout_v8` was sealed before selector hardening and executed live once.
- No engine tuning was performed after observing holdout v8.
- Metrics use structural v8 adjudication, not lexical similarity.
- Percentages include numerator and denominator whenever applicable.
- Candidate Simulation remained disabled; the predictive path was read-only.

## Metric populations

- Development: 87 total cases, 77 evaluable, 69 positive/evaluable cases.
- Holdout v8: 20 total cases, 19 evaluable, 18 recommendations and one abstention.
- Recommendation precision uses only recommended cases as its denominator.
- Coverage and false abstention use positive/evaluable cases.
- Root/repair/top-1 validity use all evaluable cases.
- Pairwise root accuracy counts structurally adjudicable root comparisons.
- Candidate pairwise accuracy counts candidate pairs with an explicit preference.

## Canonical development

| Metric | Before | After live | Delta |
|---|---:|---:|---:|
| Root-cause validity | 81.2% | 67/69 = 97.1% | +15.9 pp |
| Repair-pair validity | 89.9% | 67/69 = 97.1% | +7.2 pp |
| Top-1 validity | 89.6% | 66/69 = 95.7% | +6.1 pp |
| Recommendation precision | 89.6% | 66/68 = 97.1% | +7.5 pp |
| Recommendation coverage | 97.1% | 68/69 = 98.6% | +1.5 pp |
| False abstention | 2.9% | 1/69 = 1.4% | -1.5 pp |
| Root pairwise accuracy | n/a | 287/313 = 91.7% | n/a |
| Candidate pairwise accuracy | n/a | 26/34 = 76.5% | n/a |

The final deterministic repair-pair scoring ablation, applied to the same frozen
reasoner outputs, raised candidate pairwise accuracy from 26/34 (76.5%) to
31/34 (91.2%). It changes no Qwen output and was covered by focused unit tests.

Development Qwen telemetry: 78 calls, 2,936.9 average prompt tokens, 410.7
average completion tokens, and 40.75 seconds average request latency.

## Development robustness

| Metric | Result |
|---|---:|
| Variant root validity | 19/19 = 100% |
| Variant repair-pair validity | 19/19 = 100% |
| Variant top-1 validity | 19/19 = 100% |
| Root signature invariance | 19/19 = 100% |
| Repair invariance | 19/19 = 100% |
| Target signature invariance | 19/19 = 100% |
| Problem-wording invariance | 6/6 = 100% |
| Wrapper invariance | 1/1 = 100% |
| Counterfactual root sensitivity | 4/4 = 100% |
| Counterfactual repair sensitivity | 4/4 = 100% |
| Counterfactual target sensitivity | 3/4 = 75% |

The sampled candidate/identifier permutation check was 2/3 (66.7%). The one
difference (`SV8D-008`) preserved root kind, causal origin and repair strategy,
but alternated between two valid parameter-granularity targets on the same
causal path. Evidence order remained 3/3 invariant. This residual target alias
instability is recorded rather than tuned against the benchmark.

## Ablations

| Rule/feature | Before | After | Generalization effect | Decision |
|---|---|---|---|---|
| Lexical boundary restriction | Could constrain strategy from problem words | Disabled by default | Wording invariance reached 100% | Keep disabled |
| Runtime import SCC validation | Import wording could create false cycle roots | Runtime production SCC required | Distractor and non-runtime imports do not create roots | Keep |
| Demonstrated return contracts | Manifestation and producer contract could tie | Contract flag comes from structural return/use evidence | Root validity improved without unsupported claims | Keep |
| Semantic root dominance | Input order influenced obvious comparisons | Dominance requires compatible causal roles and path | Root pairwise accuracy reached 91.7% development | Keep |
| Strategy-only ranking | Equivalent strategy labels ignored target role | Score strategy+target pair structurally | Frozen-output pairwise 76.5% -> 91.2% | Keep |
| Raw target equality | Function/parameter/return aliases looked different | Semantic target signatures encode role and relation | Target invariance reached 100% | Keep |

## Holdout v8 one-shot

| Metric | Result | Gate | Pass |
|---|---:|---:|:---:|
| Root-cause validity | 12/19 = 63.2% | >= 82% | No |
| Repair-strategy validity | 16/19 = 84.2% | diagnostic | — |
| Repair-target validity | 16/19 = 84.2% | diagnostic | — |
| Repair-pair validity | 15/19 = 78.9% | >= 82% | No |
| Top-1 validity | 15/19 = 78.9% | >= 82% | No |
| Recommendation precision | 15/18 = 83.3% | >= 82% | Yes |
| Recommendation coverage | 18/19 = 94.7% | diagnostic | — |
| False abstention | 1/19 = 5.3% | <= 8% | Yes |
| Root pairwise accuracy | 37/83 = 44.6% | diagnostic | — |
| Candidate pairwise accuracy | 10/10 = 100% | >= 80% | Yes |
| Evidence-reference validity | 100% | 100% | Yes |
| Unsupported-claim rate | 0% | <= 5% | Yes |

Qwen telemetry: 18 calls, 2,547.7 average prompt tokens, 403.9 average
completion tokens, and 35.19 seconds average request latency.

### Holdout failure analysis

- `SV8H-004`: false abstention in an upstream producer/return flow.
- `SV8H-010`, `SV8H-011`: selected `RETURN_CONTRACT` manifestations instead
  of the adjudicated `ARGUMENT_BINDING` origin; both also caused ranking errors.
- `SV8H-014`: selected the sibling `entries` argument rather than `slots`,
  producing root, target and ranking errors.
- `SV8H-017` through `SV8H-019`: the production import cycle and safe repair
  were found, but causal identity used the imported callable instead of the
  module/SCC identity required by the structural truth.

The key result is not a scoring failure: valid repair-pair comparisons were
10/10. The dominant residual defect is root semantic identity/selection before
candidate ranking.

## Holdout v8 robustness

| Metric | Result | Gate | Pass |
|---|---:|---:|:---:|
| Variant root validity | 16/19 = 84.2% | >= 82% | Yes |
| Variant repair-pair validity | 19/19 = 100% | >= 82% | Yes |
| Variant top-1 validity | 19/19 = 100% | diagnostic | — |
| Root signature invariance | 19/19 = 100% | >= 88% | Yes |
| Repair invariance | 19/19 = 100% | >= 88% | Yes |
| Target signature invariance | 19/19 = 100% | diagnostic | — |
| Problem-wording invariance | 6/6 = 100% | >= 85% | Yes |
| Wrapper invariance | 1/1 = 100% | diagnostic | — |
| Counterfactual root sensitivity | 4/4 = 100% | diagnostic | — |
| Counterfactual repair sensitivity | 4/4 = 100% | diagnostic | — |
| Counterfactual target sensitivity | 3/4 = 75% | diagnostic | — |
| Structured/decision stability | 25/25 = 100% | diagnostic | — |

High invariance does not rescue the canonical gate: the engine repeats some
incorrect root selections consistently across superficial transformations.

## Recovery regression and seed synthesis

On the 15 inherited v7 development cases, bounded recovery remained healthy:
5 attempts, 5 successes, 5 correct recovered diagnoses, 100% precision and 0%
noise. No recovery budget or expansion rule was changed in v8.

Development seed synthesis: 13 attempts, one unique seed, one correct diagnosis
(100% precision on successful synthesis). Holdout: two attempts, no unique seed;
the service failed closed rather than guessing.

## Safety

Across development, canonical holdout and robustness holdout:

```text
filesystem_mutations                = 0
tool_dispatches                     = 0
execution_authorized                = 0
authority_grants                    = 0
destructive_actions                 = 0
forbidden_candidate_recommendations = 0
evidence_reference_validity         = 100%
unsupported_claim_rate              = 0%
```

## Gate decision

`PREDICTIVE_QUALITY_READY = false`

Canonical holdout root validity (63.2%), repair-pair validity (78.9%) and top-1
validity (78.9%) missed their 82% gates. The consultative engine must not be
frozen as quality-ready yet. The next predictive iteration should focus narrowly
on module/SCC import identity and pairwise root selection for structurally
equivalent argument/return manifestations, using a newly sealed benchmark.

`SIMULATION_RUNTIME_READY = false`

Candidate Simulation was not implemented or executed. The existing sandbox
provider remains fail-closed and no secure runtime has been demonstrated.
