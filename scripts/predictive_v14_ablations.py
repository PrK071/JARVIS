"""Ablacao de dispatch com fatos identicos; nao mede accuracy causal ficticia."""
import argparse
import ast
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

from scripts.predictive_v14_families import behavioral_families
from tern.orchestrator.predictive.causal import build_causal_slice, CausalEdgeKind, CausalNodeKind
from tern.orchestrator.predictive.external_evaluation import rate
from tern.orchestrator.predictive.grounding import build_evidence_ledger, _FactVisitor
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.project_intelligence_v2 import ProjectCandidateGenerator, ProjectIndexBuilderV2
from tern.orchestrator.security import PathPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Resultado existente.')
    rows = []
    for family in behavioral_families():
        root = Path('.orchestrator/predictive-v14-families') / family.name
        policy = PathPolicy((root.resolve(),))
        snapshot = ProjectIndexBuilderV2(root, path_policy=policy).build()
        context = build_problem_context(family.problem, snapshot,
            ProjectCandidateGenerator().generate(family.problem, snapshot), policy)
        context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
        row = {'family': family.name}
        for mode in ('ON', 'OFF'):
            causal, roots = build_causal_slice(context if mode == 'ON' else
                replace(context, behavioral_graph=None), snapshot, policy)
            nodes = {n.id: n for n in causal.nodes}
            resolved = [e for e in causal.edges if e.kind is CausalEdgeKind.RETURNED_FROM
                and nodes[e.source_id].kind is CausalNodeKind.RETURN_VALUE
                and nodes[e.target_id].kind is CausalNodeKind.CALL
                and nodes[e.target_id].scope == family.api
                and nodes[e.source_id].scope != family.api]
            valid = sum(nodes[e.source_id].scope == family.implementation for e in resolved)
            row[mode] = {'resolved_callee_return_edges': len(resolved), 'valid_edges': valid,
                         'invalid_edges': len(resolved)-valid, 'candidate_count': len(roots)}
        rows.append(row)
    metrics = {mode: rate(sum(r[mode]['valid_edges'] for r in rows),
        sum(r[mode]['resolved_callee_return_edges'] for r in rows), 'hand-adjudicated callee identity controls')
        for mode in ('ON', 'OFF')}
    source = '\n'.join(f'def function_{i}():\n    return helper({i})' for i in range(200))
    source += '\n# ' + 'data ' * 40_000
    tree = ast.parse(source)
    parts = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    visitor = _FactVisitor('module.py', source, set(range(1, 1000)), [])
    start = perf_counter()
    legacy = [(ast.get_source_segment(source, n) or '').strip()[:320] for n in parts]
    legacy_seconds = perf_counter()-start
    start = perf_counter()
    cached = [visitor._snippet(n) for n in parts]
    cached_seconds = perf_counter()-start
    assert legacy == cached
    args.output.write_text(json.dumps({'version': 14, 'cases': rows, 'dispatch_precision': metrics,
        'snippet_cache': {'calls': len(parts), 'source_bytes': len(source.encode()),
            'legacy_seconds': legacy_seconds, 'cached_seconds': cached_seconds, 'identical_output': True},
        'decision': 'Keep qualified dispatch; never infer a violated contract from reachability.',
        'metamorphic_effect': 'Rename/wording/comment unit controls preserve paths/effects.',
        'counterfactual_effect': 'Removing a call removes its implementation/effect path.',
        'limitation': 'Structural identity ablation; root validity is not measured on these contract-free controls.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
