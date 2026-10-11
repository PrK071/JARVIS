"""Profiling estatico: nunca importa nem executa o projeto analisado."""
import argparse
import cProfile
import hashlib
import json
import pstats
import time
from dataclasses import replace
from pathlib import Path

from tern.orchestrator.security import PathPolicy
from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2, ProjectCandidateGenerator
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.behavioral import bind_graph_evidence
from tern.orchestrator.predictive.causal import build_causal_slice
from tern.orchestrator.predictive.semantic import pairwise_root_matrix
from tern.orchestrator.predictive.external_evaluation import load_external_cases, python_source_hash


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--case', default='sphinx-doc__sphinx-7975')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--snapshots', type=Path, default=Path('C:/Users/User/AppData/Local/Temp/JARVIS-v11/.orchestrator/external-v15/snapshots'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Nao sobrescrever perfil.')
    cases = load_external_cases(Path('tests/data/predictive/external_v15'), split='all')
    case = next(c for c in cases if c['id'] == args.case)
    root = args.snapshots / case['id']
    before = python_source_hash(root)
    policy = PathPolicy((root.resolve(),))
    phases, profiles = {}, {}
    def measured(name, function, *arguments):
        start = time.perf_counter()
        profiler = cProfile.Profile()
        result = profiler.runcall(function, *arguments)
        phases[name] = time.perf_counter()-start
        entries = sorted(pstats.Stats(profiler).stats.items(), key=lambda item: item[1][3], reverse=True)
        profiles[name] = [dict(file=Path(k[0]).name, line=k[1], function=k[2],
            calls=v[1], own_seconds=v[2], cumulative_seconds=v[3]) for k, v in entries[:20]]
        print(f'{name}: {phases[name]:.3f}s', flush=True)
        return result
    snapshot = measured('index', ProjectIndexBuilderV2(root, path_policy=policy).build)
    selection = measured('retrieval', ProjectCandidateGenerator().generate, case['problem_formulations'][0], snapshot)
    context = measured('context', build_problem_context, case['problem_formulations'][0], snapshot, selection, policy)
    context = replace(context, evidence_ledger=measured('ledger', build_evidence_ledger, context, snapshot, policy))
    context = replace(context, behavioral_graph=bind_graph_evidence(context.behavioral_graph, context.evidence_ledger))
    slice_, roots = measured('causal_slice', build_causal_slice, context, snapshot, policy)
    measured('pairwise_comparison', pairwise_root_matrix, roots, slice_)
    if python_source_hash(root) != before:
        raise AssertionError('Projeto externo alterado.')
    serialized = json.dumps({'slice': slice_.as_dict(), 'roots': [r.as_dict() for r in roots]}, sort_keys=True).encode()
    args.output.write_text(json.dumps({'baseline': '4c259064f8761b0295ff846e45fa63ca21c61690',
        'case_id': case['id'], 'purpose': 'historical diagnostic profiling, never promotion',
        'phases': phases, 'hotspots': profiles, 'nodes': len(slice_.nodes), 'edges': len(slice_.edges),
        'roots': len(roots), 'semantic_sha256': hashlib.sha256(serialized).hexdigest(),
        'filesystem_mutations': 0, 'qwen_requests': 0}, indent=2)+'\n')


if __name__ == '__main__':
    main()
