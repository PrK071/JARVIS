"""Intervencoes controladas sao diagnosticas, nunca accuracy externa."""
import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path

from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2, ProjectCandidateGenerator
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.causal import build_causal_slice
from tern.orchestrator.predictive.behavioral import bind_graph_evidence
from tern.orchestrator.predictive.contracts import build_contract_contexts
from tern.orchestrator.predictive.sufficiency import assess_evidence_sufficiency
from tern.orchestrator.predictive.investigation import build_investigable_hypotheses
from tern.orchestrator.security import PathPolicy


def control(source, problem, root):
    root.mkdir(parents=True)
    path = root / 'api.py'
    path.write_text(source, encoding='utf-8')
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    policy = PathPolicy((root.resolve(),))
    index = ProjectIndexBuilderV2(root, path_policy=policy).build()
    selection = ProjectCandidateGenerator().generate(problem, index)
    context = build_problem_context(problem, index, selection, policy)
    context = replace(context, evidence_ledger=build_evidence_ledger(context, index, policy))
    context = replace(context, behavioral_graph=bind_graph_evidence(context.behavioral_graph, context.evidence_ledger))
    slice_, roots = build_causal_slice(context, index, policy)
    context = replace(context, causal_slice=slice_, root_cause_candidates=roots)
    contracts = build_contract_contexts(context, policy)
    assessment = assess_evidence_sufficiency(context, contracts)
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    return {'api': len(context.behavioral_graph.entry_ids), 'effects': len(context.behavioral_graph.effects),
        'roots': len(roots), 'supported_roots': len(assessment.supported_root_ids),
        'investigation_locations': len(build_investigable_hypotheses(context, contracts)),
        'violation_supported': any(c.violation_supported for c in contracts),
        'assessment': assessment.as_dict()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.workspace.exists():
        raise ValueError('Nao sobrescrever controles.')
    sources = {'unconditional_mismatch': 'def read() -> int:\n    return None\n',
        'conditional_unverified': 'def read(flag) -> int:\n    if flag:\n        return None\n    return 1\n',
        'correct_bool_subtype': 'def read() -> int:\n    return True\n',
        'reported_contract_only': 'def read(item):\n    return item\n'}
    rows = []
    for name, source in sources.items():
        a = control(source, 'Unexpected output', args.workspace / name / 'normal')
        b = control(source, 'read() unexpected output', args.workspace / name / 'oracle_api')
        rows.append({'control': name, 'normal': a, 'oracle_api_identity_only': b,
            'oracle_candidate_exists_but_unverified': bool(b['roots'] and not b['supported_roots']),
            'oracle_root_promotion_permitted': bool(b['supported_roots']),
            'intervention_provenance': 'EVALUATOR_ONLY: known API, not cause. Same source; no oracle in runtime.'})
    args.output.write_text(json.dumps({'version': 16, 'rows': rows, 'qwen_requests': 0,
        'scope': 'Controlled static diagnostic interventions, not external accuracy.',
        'contract_oracle_limitation': 'Verified violation needs actual AST evidence; unverified contract/oracle root cannot bypass promotion.',
        'external_execution': 0}, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
