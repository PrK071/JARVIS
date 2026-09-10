# Predictive grounding and ranking hardening: failure analysis

## Protocol

- BEFORE is the preserved 30-case development live result produced before the
  Evidence Ledger changes (`grounding-hardening-before.json`).
- Hardening used development only. `historical_holdout_v1` was not treated as
  unseen.
- `holdout_v2` was sealed at
  `cea2650c718a9d6214acdbf30e1d8ba7f71a2ebdbd2615d025a1012e7e936bab`
  before behavioral hardening and executed live once after the code freeze.
- All lexical metrics remain deterministic approximations; no LLM judge is used.

## Layered diagnosis before hardening

| Layer | Evidence | Diagnosis |
|---|---:|---|
| Retrieval | recall@3 98.1%, valid refs 100% | Context discovery was not the dominant failure. |
| Missing structural relation | PD-002 missed the object field definition | Instance attributes assigned through `self.<name>` were absent from the symbol index. |
| Grounding | unsupported claims 64.2%, support precision 28.4% | A valid `path:line` was treated as support for free causal prose. |
| Root cause | 12 `WRONG_ROOT_CAUSE` | The model could state a plausible cause without claim-to-fact linkage. |
| Candidate generation | solution-family hit 63.3% | Candidates inherited unsupported hypotheses and unstructured prose. |
| Family duplication | 6 duplicate failures | Text normalization did not identify the same technical strategy. |
| Forbidden candidate | 1 forbidden solution | Dangerous consultative strategies had no pre-ranking hard filter. |
| Ranking | 17 wrong rankings; hit 37.0% | Invalid candidates reached scoring and correlated model scores dominated. |
| Tie / ambiguity | ID broke ties | Determinism was mistaken for evidence of a winner. |
| Reasoner | no outage in baseline | Model unavailability already needed to remain distinct from missing evidence. |

## Development AFTER

Hardening removed invalid references and unsupported claims from eligible output,
but did not make the model's root-cause or repair-family selection reliable enough.

| Failure code | Cases | Example | Responsible component | Simulation blocker |
|---|---:|---|---|---|
| `WRONG_RANKING` | 13 | PD-003 preferred a non-acceptable repair family | scoring inputs/model ordinal judgments | yes |
| `WRONG_ROOT_CAUSE` | 10 | PD-001 described `None` correctly but missed the corpus' operand signal | reasoner plus lexical ground-truth approximation | yes |
| `WRONG_SOLUTION_FAMILY` | 7 | PD-005 produced no acceptable AttributeError strategy | candidate generation | yes |
| `UNSUPPORTED_CLAIM` | 4 | PD-002 was safely rejected after invalid claim linkage | claim mapping / reasoner contract | yes, through lost coverage |
| `NO_CANDIDATE` / false abstention | 4 | PD-011 lost all candidates after grounding validation | candidate eligibility | yes |
| `AMBIGUOUS_RANKING` | 1 | PD-005 emitted no artificial winner on a score tie | ranking margin | no by itself; behavior is desired |
| duplicate family | 0 | structured signatures removed all generated duplicates | deduplication | no |
| forbidden recommendation | 0 | policy tests reject destructive/bypass strategies | candidate policy | no |

The score correlation audit changed materially. BEFORE,
`estimated_success` and candidate `confidence` correlated at 0.986 and
`estimated_success` also correlated with evidence at 0.604. AFTER, confidence is
kept only as a compatibility/reporting field and is not a score feature;
`estimated_success` is the bounded hypothesis estimate, while evidence is derived
from ledger atom strengths. Their observed correlation is -0.122. The score
weights were not changed because development still showed no reliable ranking
improvement from a margin threshold or an alternative weighting.

Development ranking margins did not justify an arbitrary ambiguity threshold:
wrong winners occurred both near and far from ties. Only an exact top-score tie is
therefore abstained as `AMBIGUOUS_RANKING`.

## Holdout v2 (single live run; no subsequent tuning)

The unseen split preserved 100% evidence-reference validity, 0 unsupported claims,
and 0 forbidden recommendations, but exposed remaining generalization failures:

- 5 `WRONG_RANKING` cases;
- 2 false abstentions / no-candidate cases;
- 2 wrong solution-family cases;
- 1 wrong-root-cause case (PV2-008);
- 1 invalid structured model response (PV2-001);
- 1 relevant-file miss (PV2-003);
- 1 explicit ambiguous ranking (PV2-011).

Because hypothesis, solution-family, support-precision, and ranking targets were
not jointly met, this stage remains blocked. The next iteration should improve
the reasoner's concise causal-claim contract and obtain more independent ranking
signals before any Candidate Simulation is introduced.
