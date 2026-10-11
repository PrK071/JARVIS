"""Pos-analise somente leitura; nao muda casos, outputs, thresholds ou runtime."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path

from tern.orchestrator.predictive.external_evaluation import rate


def audit(report_path, archive):
    report = json.loads(report_path.read_text())
    rows = []
    claims, unsupported, refs, valid_refs = 0, 0, 0, 0
    for row in report['results']:
        if row['status'] != 'COMPLETED':
            continue
        path = archive / row['case_id'] / row['input_mode'] / 'trace.json.gz'
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != row['trace_sha256']:
            raise ValueError('Trace com hash invalido.')
        payload = json.loads(gzip.decompress(raw))
        decision = payload['decision']
        graph = decision.get('behavioral_graph') or {}
        contracts = decision.get('behavioral_contract_contexts', ())
        all_claims = [c for h in decision.get('hypotheses', ()) for c in h.get('claims', ())]
        claims += len(all_claims)
        unsupported += sum(c['support'] == 'UNSUPPORTED' for c in all_claims)
        refs += row['evidence_reference_population']
        valid_refs += row['evidence_references_valid']
        rows.append({'case_id': row['case_id'], 'mode': row['input_mode'],
            'report_available': bool(decision['problem']), 'reported_api_resolved': bool(row['state']['api_ids']),
            'implementation_reachable': bool(graph.get('callables')),
            'effects_represented': bool(row['state']['effect_ids']),
            'contract_exists': any(c['contract_exists'] for c in contracts),
            'contract_applies': any(c['contract_applies'] for c in contracts),
            'violation_supported': any(c['violation_supported'] for c in contracts),
            'roots_present': bool(row['state']['root_ids']),
            'roots_supported': bool(row['state']['supported_root_ids']),
            'unknown_expectation_sentinel': any(c.get('expected_behavior') == 'UNKNOWN' for c in contracts),
            'outcome': 'DIAGNOSIS' if row['recommended'] else 'INVESTIGATION' if row['hypotheses_emitted'] else 'ABSTENTION',
            'structural_edges_do_not_prove_execution': True})
    primary = [r for r in rows if r['mode'] == 'A']
    funnel = {field: rate(sum(r[field] for r in primary), len(primary), 'completed independent defects in A')
        for field in ('report_available', 'reported_api_resolved', 'implementation_reachable', 'effects_represented',
            'contract_exists', 'contract_applies', 'violation_supported', 'roots_present', 'roots_supported')}
    funnel['affected_api_recall'] = rate(0, 0, 'undefined: no independently sealed affected-API oracle; resolution is not recall')
    funnel['causal_bridge_representability'] = report['summary']['A']['metrics']['correct_candidate_exists']
    funnel['root_selection_given_correct_candidate'] = rate(0, 0, 'undefined: no correct-root candidates available')
    primary_results = [r for r in report['results'] if r['input_mode'] == 'A' and r['status'] == 'COMPLETED']
    funnel['recommendation_coverage'] = rate(sum(r['recommended'] for r in primary_results), len(primary_results), 'completed independent A defects')
    funnel['root_support_validity'] = rate(0, 0, 'undefined: no independent semantic support adjudication of every generic root')
    return {'source': str(report_path), 'funnel': funnel,
        'routing': dict(Counter(r['outcome'] for r in primary)), 'rows': rows,
        'evidence_reference_validity': rate(valid_refs, refs, 'candidate/investigation evidence IDs, not causal correctness'),
        'unsupported_claims': unsupported, 'claim_population': claims,
        'unsupported_claim_rate': rate(unsupported, claims, 'claims; hypothesis-only suggestions are not confirmed claims'),
        'limitations': ['A/B inputs are correlated formulations, not additional independent defects.',
            'B UNKNOWN sentinel can create unverified reported-contract metadata; new IDs are not evidence of information gain.',
            'Hypothesis location precision is exact root-location proxy; it can penalize a legitimate investigation at the public API.',
            'Two class-scoped oracles are coarse; exact method/attribute correspondence is not fully adjudicated.',
            'Requests includes historical Python-2 behavior; not a qualification of arbitrary modern Python environments.',
            'No additional runtime observations were available; investigation requests were not independently validated for efficacy.']}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results', type=Path, default=Path('tests/data/predictive/results'))
    parser.add_argument('--archive', type=Path, default=Path('D:/JARVIS-evaluation-v16/artifacts/v16'))
    args = parser.parse_args()
    output = args.results / 'v16-evidence-study-audit.json'
    with output.open('x', encoding='utf-8') as f:
        json.dump({'version': 16, 'audits': [audit(args.results / filename, args.archive / directory)
            for filename, directory in (('v16-development-before.json', 'development-before'),
                ('v16-development-after.json', 'development-after'),
                ('v16-external-holdout-one-shot.json', 'holdout-one-shot'))]}, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    main()
