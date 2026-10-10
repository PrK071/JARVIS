# v14 diagnosis before runtime changes

Frozen engine: b00a4813278699735d77c41a593cea99790ff8f7.
Historical v12/v13 results are diagnostic, not a fresh holdout.

| Mechanism | Evidence in source | Consequence |
|---|---|---|
| Existing line anchor suppresses symbol enrichment | service.build_problem_context checks only absence of _line_target | Anchor existence is mistaken for sufficiency |
| Callable resolution drops owner/receiver | symbols.resolved_callees excludes methods and relative imports | self helper and relative helper paths disappear |
| Unique leaf matching in causal builder | causal.build_causal_slice builds by_name from final name | Reachability may connect a dynamic receiver to unrelated callable |
| Return/condition-only behavioral fallback | causal seeding restricts API nodes to RETURN_VALUE/CONDITION | Void state changes and iteration cannot anchor behavior |
| No typed effect transport | grounding._FactVisitor lacks yield and handler facts | Symbol presence cannot represent operation/effect relationship |
| Same-file fallback evidence | causal._evidence_at uses first two path atoms when no local match | Existing nodes do not prove the necessary edge |

v13's oracle-symbol intervention produced only 1/6 localized candidates.
Adding source AST atoms alone produced 0/6. These observations rule out
symbol-window coverage as a sufficient abstraction. They do not prove the
selector is wrong in the five cases without representable candidate sets.

The smallest safe intervention is an additive bounded implementation graph,
with exact call-site evidence and explicit unresolved dispatch. Calls and
effects are structural observations, not defect assertions. An observation
anchor with no supported contract must remain NO_SUPPORTED_CONTRACT.
Do not create a new root simply because an API reaches a helper.

Keep scoring, v7 recovery, relevance gate, policy and sandbox unchanged.
Measure paths/effects on positive and negative development families before
integrating source windows; preserve original windows and anchor diagnostics.
