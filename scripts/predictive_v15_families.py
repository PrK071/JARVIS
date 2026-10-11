"""Baseline e familias locais confiaveis; nenhum codigo do fixture executado."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from tern.orchestrator.predictive.behavioral import resolve_behavioral_context, bind_graph_evidence
from tern.orchestrator.predictive.causal import build_causal_slice
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.models import ProblemContext, EvidenceExcerpt
from tern.orchestrator.predictive.testing.contract_families import contract_families
from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2
from tern.orchestrator.security import PathPolicy


def evaluate(workspace, *, contracts=False):
    rows = []
    for family in contract_families():
        root = workspace / family.name
        root.mkdir(parents=True, exist_ok=True)
        path = root / 'api.py'
        if path.exists() and path.read_text() != family.source:
            raise ValueError('Fixture divergente.')
        path.write_text(family.source, encoding='utf-8')
        policy = PathPolicy((root,))
        snapshot = ProjectIndexBuilderV2(root, path_policy=policy).build()
        context = ProblemContext(family.problem, str(root), family.name,
            (EvidenceExcerpt('api.py:1-1', 'api.py', 1, 1, 'STRONG', (), family.source.splitlines()[0]),),
            ('api.py',), (), ())
        context = resolve_behavioral_context(context, snapshot, policy)
        ledger = build_evidence_ledger(context, snapshot, policy)
        context = replace(context, evidence_ledger=ledger,
            behavioral_graph=bind_graph_evidence(context.behavioral_graph, ledger))
        causal, roots = build_causal_slice(context, snapshot, policy)
        row = {'family': family.name, 'expected': family.expected,
            'api_anchored': bool(context.behavioral_graph.entry_ids),
            'effect_available': bool(context.behavioral_graph.effects),
            'root_candidate_count': len(roots),
            'violation_supported': None, 'investigation_symbols': [],
            'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        if contracts:
            from tern.orchestrator.predictive.contracts import build_contract_contexts
            from tern.orchestrator.predictive.investigation import build_investigable_hypotheses
            contexts = build_contract_contexts(context, policy)
            hypotheses = build_investigable_hypotheses(context, contexts)
            row.update(violation_supported=any(c.violation_supported for c in contexts),
                investigation_symbols=[h.symbol for h in hypotheses],
                contracts=[c.as_dict() for c in contexts])
        rows.append(row)
    return {'version': 15, 'population': '12 controlled local families, not external accuracy',
        'rows': rows, 'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path('tern/orchestrator/predictive').glob('*.py'))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--contracts', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Nao sobrescrever historico.')
    value = evaluate(args.workspace.resolve(), contracts=args.contracts)
    args.output.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'families': len(value['rows']), 'roots': sum(bool(r['root_candidate_count']) for r in value['rows'])}))


if __name__ == '__main__':
    main()
