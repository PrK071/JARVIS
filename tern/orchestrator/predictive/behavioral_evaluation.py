"""Intervencoes experimentais. Nunca importadas pelo runtime consultivo."""
from dataclasses import replace

from .behavioral import bind_graph_evidence, resolve_behavioral_context
from .causal import build_causal_slice
from .coverage_evaluation import layer_coverage
from .external_evaluation import matches_origin
from .grounding import build_evidence_ledger
from .oracle_coverage_evaluation import insert_oracle_symbol


def behavioral_interventions(context, snapshot, policy, oracle):
    def observe(item):
        ledger = build_evidence_ledger(item, snapshot, policy)
        item = replace(item, evidence_ledger=ledger,
            behavioral_graph=bind_graph_evidence(item.behavioral_graph, ledger))
        causal, roots = build_causal_slice(item, snapshot, policy)
        return item, roots, layer_coverage(snapshot, item, causal, roots, oracle)

    a, _, checks_a = observe(context)
    experiments = {'A_NORMAL': {'status': 'OBSERVED', 'coverage': checks_a}}
    api = oracle.get('affected_api')
    if api:
        # Supplying the observed API identity is not supplying the corrected
        # implementation symbol or a stronger behavioral contract.
        b = resolve_behavioral_context(replace(context, problem=context.problem + f'\nAPI: `{api}`'),
                                       snapshot, policy)
        b, _, checks_b = observe(replace(b, problem=context.problem))
        experiments['B_KNOWN_API'] = {'status': 'OBSERVED', 'coverage': checks_b}
    else:
        experiments['B_KNOWN_API'] = {'status': 'API_NOT_INDEPENDENTLY_IDENTIFIED'}
    # Do not relabel normal resolver edges as oracle-injected paths. A known
    # corrected file alone cannot adjudicate a dynamic API dispatch edge.
    experiments['C_KNOWN_IMPLEMENTATION_PATH'] = {
        'status': 'NOT_ADJUDICATED',
        'reason': 'Upstream correction site does not establish a complete API dispatch path.'}
    effect_context = insert_oracle_symbol(context, snapshot, policy, oracle)
    effect_context = resolve_behavioral_context(replace(effect_context,
        problem=f"`{oracle['root_symbol']}`"), snapshot, policy)
    effect_context, roots, checks_d = observe(replace(effect_context, problem=context.problem))
    matching_effects = [e for e in effect_context.behavioral_graph.effects
        if e.path == oracle['root_path'] and e.owner == oracle['root_symbol']
        and e.kind.value == oracle.get('behavioral_effect') and e.evidence_ids]
    experiments['D_KNOWN_EFFECT'] = {'status': 'OBSERVED' if matching_effects else 'EFFECT_NOT_REPRESENTABLE',
        'effect_ids': [e.id for e in matching_effects], 'coverage': checks_d,
        'limitation': 'Effect owner/window is oracle-provided; never a real accuracy observation.'}
    valid = [r for r in roots if matches_origin(r.as_dict(), oracle)]
    experiments['E_KNOWN_CAUSAL_CANDIDATE'] = {
        'status': 'GROUNDABLE_CANDIDATE_AVAILABLE_NOT_SELECTED' if valid else 'ORACLE_ROOT_NOT_REPRESENTABLE',
        'candidate_ids': [r.id for r in valid],
        'selection_accuracy': None,
        'reason': 'No selection call in this offline intervention study; absent contracts are never fabricated.'}
    return {'version': 14, 'experimental_only': True, 'experiments': experiments,
            'qwen_calls': 0, 'gold_diff_in_input': False}
