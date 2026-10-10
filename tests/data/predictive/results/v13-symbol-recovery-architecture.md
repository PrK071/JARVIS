# v13 bounded symbol context

The runtime reuses the existing project index, ledger, causal slice and root
models. It does not consult external adjudication, corrective patches or an
oracle. Ranking weights, candidate policy, v7 recovery budget and v12 SCC
relevance gate are unchanged.

## Transport, not defect proof

Reported code-shaped API identifiers are resolved against definitions in
already retrieved files. Exact qualified matches are observational anchors;
receiver-method matches retain every possible owner in that scope. Exact
AST reads can locate a reported configuration/value reference. Comments,
docstrings and string-literal mentions are not structural reads.

One bounded expansion follows uniquely resolved local/module-import calls,
including aliases. Shadowed imports and dynamic receiver dispatch remain
unresolved. Limits: one attempt, two extra files, sixteen symbols, depth two,
240 lines per symbol and 48,000 additional UTF-8 context bytes. These are
additional budgets, not a claim that the legacy initial context is bounded
to those numbers. A pre-existing ambiguity in basename matching can still
retrieve many `__init__.py` files; v13 does not expand its retrieval top-k.

Each RelevantSymbolContext records qualified owner, source range, selection
reason, anchors, actual calls, returns, assignments and annotation presence.
`SYMBOL_RETRIEVED_NOT_PROVEN_DEFECTIVE` is explicit. An annotation is not
proof that its contract was violated. Facts outside a truncated window are
not advertised as added context.

The ledger unions disjoint windows per file, parses each selected source
once, and preserves qualified class/function ownership. Existing fact kinds
are reused; no fabricated semantic atoms are introduced. Lookup indexes
preserve atom IDs, order and `_evidence_at` fallback semantics.

An affected API can provide an inferred behavioral symptom anchor only when
the existing pipeline has no usable anchor and the selected API is unique.
It uses return/condition nodes, never a fabricated exception. The bridge is
non-deterministic: reported API involvement is not demonstrated defect origin.
Existing causal anchors take precedence, after a synthetic regression exposed
that replacing one could change parameter-owner attribution.

## Public and evaluation boundaries

DecisionReport keeps schema version 6 and adds symbol_context_schema_version
1 plus relevant_symbol_contexts. Old fields and command interfaces remain.
The runtime has no writer, ToolRegistry, executor or patch facility.

Coverage distinguishes indexed, windowed, ledger definition, local facts,
local AST relation proxy, generated root and selected root. Window coverage
and explicit symbol context are combined. Final selection uses final recovered
root IDs, not initial IDs. A local relation is not proof of the adjudicated
defect-bearing relation. Location correctness is not full semantic entailment.

Oracle A/B/C/D interventions are evaluation-only and never pooled with actual
accuracy. B adds the adjudicated symbol window; C adds actual AST facts, not
a narrative or gold patch. D requires a representable grounded root and is
unavailable rather than fabricating a missing path. The initial six-case
offline study predates the anchor-precedence regression fix; its diagnostic
results must not be treated as final-runtime selector accuracy.

Read-only evaluator workers have an optional per-case wall-clock limit.
Only JARVIS's own analysis runs on the host; external projects are parsed,
never imported or tested. Operationally incomplete cases are explicitly
excluded from quality denominators and block readiness. Completed sidecars
and all earlier raw traces are retained. Profiling times are inclusive and
can overlap; they must not be summed as exclusive phases.
