"""Oracle-assisted coverage diagnostics; never imported by predictive runtime."""
from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

from .external_evaluation import matches_origin, rate
from .models import EvidenceKind, ProblemContext


def layer_coverage(snapshot, context: ProblemContext, causal_slice, roots,
                   oracle: Mapping[str, Any], *, decision=None) -> dict[str, Any]:
    path, name = oracle["root_path"], oracle["root_symbol"]
    indexed = snapshot.file_index.get(path)
    symbols = [symbol for symbol in indexed.symbols if symbol.qualified_name == name
               or symbol.qualified_name.endswith("." + name)] if indexed else []
    ranges = oracle["source_locations"]

    def inside(line):
        return any(site["line"] <= line <= site["end_line"] for site in ranges)

    selected = getattr(context, "relevant_symbol_contexts", ())
    explicit = [item for item in selected if item.path == path and item.qualified_name == name]
    # v12 has no explicit symbol context: its chosen source windows are the
    # selection contract. A surrounding class definition is not a method hit.
    windows = [item for item in context.evidence if item.path == path]
    selected_hit = bool(explicit) or any(
        item.start_line <= symbol.line <= item.end_line for item in windows for symbol in symbols)
    facts = [atom for atom in context.evidence_ledger.atoms
             if atom.path == path and inside(atom.start_line)
             and atom.kind is not EvidenceKind.SYMBOL_DEFINITION]
    definitions = [atom for atom in context.evidence_ledger.atoms
                   if atom.path == path and atom.kind is EvidenceKind.SYMBOL_DEFINITION
                   and atom.symbol == name]
    correct_nodes = {node.id for node in causal_slice.nodes
                     if node.path == path and inside(node.line)
                     and (node.scope == name or node.symbol == name)}
    correct_roots = [item for item in roots if matches_origin(item.as_dict(), oracle)]
    valid_ids = {item.id for item in correct_roots}
    # Final selection may use candidates rebuilt by structural recovery.
    final_valid_ids = ({item.id for item in decision.root_cause_candidates
                        if matches_origin(item.as_dict(), oracle)} if decision else valid_ids)
    selected_ids = {item.root_cause_id for item in decision.hypotheses} if decision else set()
    checks = {
        "correct_file_in_project_index": indexed is not None,
        "correct_symbol_in_project_index": bool(symbols),
        "correct_file_retrieved": path in context.related_files,
        "correct_symbol_selected": selected_hit,
        "correct_symbol_in_context": selected_hit,
        "correct_symbol_in_ledger": bool(definitions),
        "correct_symbol_facts_in_ledger": bool(facts),
        "correct_relation_in_causal_slice": any(
            edge.source_id in correct_nodes and edge.target_id in correct_nodes
            for edge in causal_slice.edges),
        "correct_root_candidate_generated": bool(correct_roots),
        "correct_root_selected": bool(selected_ids) and selected_ids <= final_valid_ids if decision else None,
        "correct_repair_generated": any(
            candidate.root_cause_id in final_valid_ids and candidate.strategy_kind in oracle["repair_strategies"]
            and candidate.repair_targets
            and all(target.path in oracle["target_paths"] for target in candidate.repair_targets)
            for candidate in decision.candidates) if decision else None,
    }
    stage_codes = {
        "correct_file_in_project_index": "FILE_NOT_INDEXED",
        "correct_symbol_in_project_index": "SYMBOL_NOT_INDEXED",
        "correct_file_retrieved": "FILE_NOT_RETRIEVED",
        "correct_symbol_selected": "SYMBOL_WINDOW_MISSED",
        "correct_symbol_in_ledger": "SYMBOL_DEFINITION_NOT_MATERIALIZED",
        "correct_symbol_facts_in_ledger": "SYMBOL_FACTS_NOT_MATERIALIZED",
        "correct_relation_in_causal_slice": "LOCAL_CAUSAL_RELATION_NOT_REPRESENTED",
        "correct_root_candidate_generated": "CORRECT_ROOT_NOT_GENERATED",
        "correct_root_selected": "CORRECT_ROOT_NOT_SELECTED",
        "correct_repair_generated": "CORRECT_REPAIR_NOT_GENERATED",
    }
    behavioral = behavioral_coverage(context, oracle)
    return {"stages": checks, "behavioral_coverage": behavioral,
            "first_loss": next((code for key, code in stage_codes.items() if checks[key] is False), None),
            "reason_codes": [code for key, code in stage_codes.items() if checks[key] is False],
            "indexed_symbols": [symbol.as_dict() for symbol in symbols],
            "selected_windows": [{"start_line": item.start_line, "end_line": item.end_line} for item in windows],
            "local_fact_ids": [atom.id for atom in facts],
            "local_causal_node_ids": sorted(correct_nodes),
            "correct_candidate_ids": sorted(valid_ids),
            "relation_metric": "local AST edge coverage proxy; not proof of the adjudicated defect relation"}


def behavioral_coverage(context, oracle):
    """Oracle compares identities; runtime never imports this evaluator."""
    graph = context.behavioral_graph
    if graph is None:
        return {"status": "NOT_INSTRUMENTED"}
    matches = [c for c in graph.callables if c.path == oracle['root_path']
               and c.qualified_name == oracle['root_symbol']]
    ids = {c.id for c in matches}
    effects = [e for e in graph.effects if e.callable_id in ids
               and e.kind.value == oracle.get('behavioral_effect')]
    api = oracle.get('affected_api')
    api_ids = {c.id for c in graph.callables if c.id in graph.entry_ids
               and api and (c.qualified_name == api or c.qualified_name.endswith('.' + api))}
    api_hit = bool(api_ids) if api else None
    reachable = set(api_ids)
    for _ in range(len(graph.callables)):
        reached = {r.target_id for r in graph.relations if r.source_id in reachable}
        if reached <= reachable:
            break
        reachable.update(reached)
    return {"status": "OBSERVED", "correct_symbol_available": bool(matches),
        "correct_api_available": api_hit,
        "implementation_path_available": bool(ids & reachable) if api else None,
        "reported_symbol_to_implementation_proxy": bool(matches),
        "behavioral_effect_available": bool(effects),
        # A source-operation match is not a represented violated contract.
        "defect_relation_representable": None,
        "matched_callable_ids": sorted(ids), "matched_effect_ids": [e.id for e in effects],
        "resolved_relations": len(graph.relations),
        "ledger_backed_relations": sum(bool(r.evidence_ids) for r in graph.relations),
        "unresolved_dispatches": len(graph.unresolved), "reason_codes": list(graph.reason_codes),
        "limitation": "Reachability/effect recall; defect representability needs independent contract-edge adjudication."}


def summarize_layer_coverage(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    layers = [row["layer_coverage"]["stages"] for row in rows]
    keys = list(layers[0]) if layers else []
    metrics = {key: rate(sum(item[key] is True for item in layers),
                         sum(item[key] is not None for item in layers),
                         "formulations with stage observed; abstentions retained") for key in keys}
    for label, before, after in (
        ("file_to_symbol_conversion", "correct_file_retrieved", "correct_symbol_selected"),
        ("symbol_to_ledger_conversion", "correct_symbol_selected", "correct_symbol_in_ledger"),
        ("ledger_to_candidate_conversion", "correct_symbol_facts_in_ledger", "correct_root_candidate_generated"),
    ):
        metrics[label] = rate(sum(item[before] and item[after] for item in layers),
                              sum(bool(item[before]) for item in layers),
                              "formulations passing source stage; local fact/location proxies")
    behavioral = [row['layer_coverage']['behavioral_coverage'] for row in rows
                  if row['layer_coverage'].get('behavioral_coverage', {}).get('status') == 'OBSERVED']
    behavior_metrics = {key: rate(sum(item[key] is True for item in behavioral),
        sum(item[key] is not None for item in behavioral), "formulations with behavioral stage observed")
        for key in ('correct_api_available', 'implementation_path_available',
                    'behavioral_effect_available', 'defect_relation_representable')}
    return {"metrics": metrics, "behavioral_metrics": behavior_metrics, "first_loss_counts": dict(sorted(Counter(
        row["layer_coverage"]["first_loss"] or "NO_OBSERVED_LOSS" for row in rows).items()))}
