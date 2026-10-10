"""Evaluation-only interventions. Never imported by the predictive service.

Location oracle is not an entailment oracle. Missing causal paths are not
fabricated to make an injected root appear grounded.
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .analysis import PredictiveAnalyzer
from .causal import build_causal_slice
from .coverage_evaluation import layer_coverage
from .external_evaluation import matches_origin
from .grounding import build_evidence_ledger
from .models import EvidenceExcerpt, EvidenceLedger
from .symbols import RelevantSymbolContext


def insert_oracle_symbol(context, snapshot, policy, oracle):
    path, name = oracle["root_path"], oracle["root_symbol"]
    symbol = next((item for item in snapshot.file_index[path].symbols
                   if item.qualified_name == name), None)
    if symbol is None:
        raise ValueError("Simbolo oracle ausente no indice.")
    source = policy.resolve(str(Path(snapshot.project_path) / path)).read_text(encoding="utf-8", errors="replace").splitlines()
    window = EvidenceExcerpt(f"{path}:{symbol.line}-{symbol.end_line}", path, symbol.line,
                            symbol.end_line, "STRONG", ("EVALUATION_ORACLE_SYMBOL",),
                            "\n".join(f"{line}: {source[line-1]}" for line in range(symbol.line, symbol.end_line + 1)))
    injected = RelevantSymbolContext("ORACLE_SYMBOL", path, name, symbol.kind.value,
                                     symbol.line, symbol.end_line, "EVALUATION_ORACLE_SYMBOL",
                                     ("INDEPENDENT_ADJUDICATION",))
    return replace(context, evidence=(*context.evidence, window),
                   related_files=tuple(dict.fromkeys((*context.related_files, path))),
                   relevant_symbol_contexts=(*context.relevant_symbol_contexts, injected))


def oracle_interventions(context, snapshot, policy, oracle, *, reasoner=None):
    """Return disjoint A/B/C/D experiments; never pooled with real accuracy."""
    symbolic = insert_oracle_symbol(context, snapshot, policy, oracle)
    symbolic = replace(symbolic, evidence_ledger=build_evidence_ledger(symbolic, snapshot, policy))
    # C adds real AST facts only, not an oracle-generated narrative, contract or
    # affected-API identity. This isolates evidence from symbol/seed selection.
    sites = oracle["source_locations"]
    extra_atoms = tuple(atom for atom in symbolic.evidence_ledger.atoms
                        if atom.path == oracle["root_path"]
                        and any(site["line"] <= atom.start_line <= site["end_line"] for site in sites))
    merged = {atom.id: atom for atom in (*context.evidence_ledger.atoms, *extra_atoms)}
    factual = replace(context, evidence_ledger=EvidenceLedger(tuple(sorted(merged.values(), key=lambda atom: atom.id))))
    results = []
    states = {}
    for mode, prepared in (("A_NORMAL", context), ("B_ORACLE_SYMBOL", symbolic), ("C_ORACLE_AST_FACTS", factual)):
        causal, roots = build_causal_slice(prepared, snapshot, policy)
        states[mode] = (prepared, causal, roots)
        results.append({"mode": mode, "experimental": mode != "A_NORMAL",
                        "layer_coverage": layer_coverage(snapshot, prepared, causal, roots, oracle),
                        "root_count": len(roots), "qwen_requests": 0,
                        "inserted_fact_ids": [atom.id for atom in extra_atoms] if mode == "C_ORACLE_AST_FACTS" else []})
    # Existing v10 seam supplies a correctly localized *grounded* root to the
    # normal selector. If representation cannot produce it even with oracle
    # symbols, report NOT_REPRESENTABLE instead of inventing causal edges.
    prepared, causal, roots = states["B_ORACLE_SYMBOL"]
    matching = tuple(root for root in roots if matches_origin(root.as_dict(), oracle))
    row = {"mode": "D_ORACLE_ROOT", "experimental": True,
           "oracle_candidate_ids": [root.id for root in matching],
           "status": "READY" if matching else "ORACLE_ROOT_NOT_REPRESENTABLE",
           "selected_correct": None, "qwen_requests": 0}
    if matching and reasoner:
        analysis = PredictiveAnalyzer(reasoner).analyze_with_diagnostics(
            replace(prepared, causal_slice=causal, root_cause_candidates=matching))
        ids = {root.id for root in matching}
        selected = {hypothesis.root_cause_id for hypothesis in analysis.hypotheses}
        row.update(status=analysis.failure_reason.value if analysis.failure_reason else "EVALUATED",
                   selected_correct=bool(selected) and selected <= ids,
                   qwen_requests=1,
                   candidates=[candidate.as_dict() for candidate in analysis.candidates])
    results.append(row)
    return {"schema_version": 13, "experimental_only": True,
            "not_in_readiness_accuracy": True, "experiments": results,
            "limitation": "D requires a representable grounded candidate. Missing paths/contracts are not fabricated. C inserts source-backed AST facts, never the gold diff."}
