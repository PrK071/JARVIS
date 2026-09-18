# Causal Identity & Responsibility v9 — Failure Analysis Before

Engine: `eb8feb6c26092267fb8427158d00baa0d9acbc81`

This audit was completed before changing `causal.py`, `semantic.py`, `repair.py`
or the selector. It uses fixture source, AST relations, v8 adjudication and the
sealed v8 reports; the engine response was not used as ground truth.

| Failure class | Cases | Correct candidate existed? | Correct entity existed? | Primary defect |
|---|---:|---|---|---|
| Import-cycle identity | 3 | The production edges existed | SCC existed, but only edge/callable candidates were exposed | Identity/generation |
| Argument vs return | 2 | Yes | Yes | Responsibility selection |
| Sibling argument | 1 | Both parameter nodes existed | Slot identity was discarded after binding | Identity/target binding |
| Upstream abstention | 1 | Incomplete final selection | Producer path existed | Selection/abstention |
| Ranking | 3 | Valid alternatives existed | Upstream root/target was already wrong | Downstream only |

## Import-cycle finding

`find_import_cycles()` already computes production-only SCCs and retains
observer edges. `build_causal_slice()` then materializes one ordinary IMPORT
node per production edge. Its symbol is the imported callable, so the root
identity becomes `parser/right.py::parse_left` rather than the causal entity
`SCC(parser.left, parser.right)`. The repair is often valid despite the root
identity mismatch.

## Argument/return finding

Call resolution maps positional and keyword arguments to formal parameters,
but `PASSED_AS_ARGUMENT` stores only source and target node IDs. Ordinal,
keyword, call-site and callee identity are lost. A later selector can prove
that some value reached a parameter, but cannot always prove which sibling
slot owned the defect.

`RETURN_CONTRACT` is represented by a return node. Contract demonstration may
still be strengthened by problem wording. This can beat a complete argument
binding even when the return merely manifests a bad input.

## Frozen layers

The audit found no independent defect in recovery, grounding, candidate policy,
general scoring weights or candidate pairwise ranking. Those layers remain
frozen for v9.
