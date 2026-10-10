# v13 diagnosis before engine hardening

Engine baseline: `a0a47093fe62e0e8875367a8e2915e1ec8cb67fe`.
No runtime file changed while this diagnosis and the new corpus were prepared.

1. ProjectIndexBuilderV2 records qualified definitions, source ranges, imports
   and referenced names. Symbol definitions are not automatically context.
2. ProjectCandidateGenerator budgets 12 files and 48 KB of context. It selects
   ONE slice per file, choosing the earliest matching definition if several
   APIs are mentioned; otherwise it starts at the beginning of the file.
   This is a file candidate API, not an explicit relevant-symbol API.
3. build_problem_context preserves that single window (or a traceback +/-12
   lines). Its range intersection misses a function enclosing the whole window.
   There is no representation of why a particular symbol entered context.
4. build_evidence_ledger keys excerpts by path. Multiple snippets for the same
   file would silently collapse to the last one. Definitions only enter when
   their start line is inside the snippet. AST facts are similarly windowed.
   Function visitor ownership uses an unqualified method name, while the index
   stores qualified class ownership. An unrelated class definition with a broad
   source range can appear supportive without covering the method's facts.
5. Ledger facts include assignments, calls, conditions, returns, imports and
   test assertions. Generator/yield output, mutation semantics, exception
   handler coverage and receiver/type-dependent method dispatch remain limited.
   Adding a definition alone cannot model these missing semantics.
6. Causal construction parses selected files plus direct import neighbors, but
   node evidence still depends on the narrow ledger windows. Method resolution
   uses unique final callable names, which is ambiguous in real libraries.
7. Without a reported location, causal seeding matches names in the problem to
   return/condition nodes and chooses the lexically first location. This cannot
   express an explicitly reported API independently of generic prose nouns.
   No candidate exists if there is no matched return or condition. CALL nodes
   are not roots, so void mutation APIs can remain unrepresentable.
8. The v12 relevance gate correctly blocks unrelated SCCs. Keep it unchanged;
   symbol selection is not proof of a defect, and a behavioral API observation
   must not be relabeled an exception or strong contract.

## Narrow proposed intervention

Resolve explicitly reported API/traceback anchors against indexed qualified
symbols in retrieved files; retain ambiguity. Bounded structural neighbors may
add at most two files and sixteen symbols in one attempt with depth two.
Materialize disjoint symbol windows without dropping existing windows. Reuse
the existing AST fact types and derive a behavioral symptom anchor only from
the selected affected API. Keep contract sufficiency unknown unless already
supported by AST/test facts. No score, policy, recovery-budget or SCC-gate tuning.

The ten new independently adjudicated defects (six development, four sealed
holdout) are from pytest, Sphinx and Flask. Their corrections never enter model
inputs. Historical v12 cases are diagnostic only. Coverage metrics distinguish
definition presence, local facts, local relation coverage and candidate origin;
none is silently equated with semantic correctness.
