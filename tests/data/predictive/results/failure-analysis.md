# Predictive evaluation v1 failure analysis

## Development retrieval

`PD-002` was the only retrieval failure. `pkg/calc.py` and its test were found,
but `pkg/orders.py` was not. The relationship is an instance attribute
(`Order.tax`) rather than an import/call/symbol edge represented by Project
Intelligence V2. This remains a Project Intelligence backlog item; predictive
did not add a second AST/index.

## Trivial baseline

The first-strong-evidence comparator passed 3/30 development cases. It had no
solution-family or ranking hits. This is expected and establishes that file
retrieval alone does not provide comparative technical reasoning.

## Development live

An initial smoke without a running server correctly produced
`REASONER_UNAVAILABLE`, distinct from structural insufficiency. The complete
development run then executed 30 cases with 54 Qwen requests, averaging 46.9
seconds per request.

Only 5/30 cases passed all deterministic checks. Valid evidence references stayed
at 100%, but unsupported claim rate was 64.2%, hypothesis hit rate 50.0%, solution
family hit rate 63.3%, and ranking hit rate 37.0%. `UNSUPPORTED_HYPOTHESIS` (22),
`WRONG_RANKING` (17), and `WRONG_ROOT_CAUSE` (12) dominated failures.

In `PD-001`, Qwen produced plausible solutions, but all three hypotheses asserted
facts about `order` construction absent from supplied evidence. Two candidates
also belonged to the same `fix_upstream` family. `PD-026` proposed falling back to
the lexical legacy distractor and was correctly classified `FORBIDDEN_SOLUTION`.

The score audit also found C1 and C2 tied at 0.8495 because they shared every
scored feature. Stable ID ordering selected C1, demonstrating that deterministic
tie behavior is not evidence that the winner is technically superior.

## Test support

Development structural test precision was 90%; holdout was 70%. Cases tagged
`related_test_not_relevant` show the remaining limitation: Project Intelligence
can prove a production-to-test relation, but not that an assertion covers the
candidate's behavior. The product now awards support only for the specific
production-file/test relationship referenced by a candidate, but deliberately
does not claim behavioral relevance.

## Stage gate

The one-time live holdout ran after development hardening, without subsequent
tuning. It passed 1/10 cases: hypothesis hit was 70.0%, solution family hit 80.0%,
ranking hit 44.4%, and unsupported claim rate 37.0%. Six holdout cases contained
duplicate solution families.

Safety and reference validity passed on both splits. Promotion is blocked by
unsupported claims and ranking quality. Estimated success was strongly correlated
with candidate confidence (0.986 on development), confirming the suspected signal
coupling, but no score weights were changed without a causal ranking experiment.
