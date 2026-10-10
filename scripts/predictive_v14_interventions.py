"""Estudo isolado com timeout; somente leitura/AST, sem codigo de terceiros."""
import argparse
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from scripts.predictive_external_eval import _communicate_bounded, EvaluationMemoryLimit
from tern.orchestrator.predictive.behavioral_evaluation import behavioral_interventions
from tern.orchestrator.predictive.external_evaluation import load_external_cases, python_source_hash, snapshot_inventory
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.project_intelligence_v2 import ProjectCandidateGenerator, ProjectIndexBuilderV2
from tern.orchestrator.security import PathPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case-id')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Resultado existente.')
    cases = load_external_cases(Path('tests/data/predictive/external_v14'), split='development')
    archive = Path('.orchestrator/predictive-artifacts/v14/interventions')
    archive.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in cases:
        if args.case_id and args.case_id != case['id']:
            continue
        if not args.case_id:
            output = archive / (case['id'] + '.json')
            command = [sys.executable, '-m', 'scripts.predictive_v14_interventions',
                       '--output', str(output), '--case-id', case['id']]
            worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                      text=True, encoding='utf-8', errors='replace')
            try:
                _, stderr = _communicate_bounded(worker, command, 120, 1024)
                if worker.returncode or not output.exists():
                    rows.append({'id': case['id'], 'status': 'INTERVENTION_INFRASTRUCTURE_ERROR',
                                 'diagnostic': stderr[-1200:]})
                else:
                    rows.extend(json.loads(output.read_text())['results'])
            except (subprocess.TimeoutExpired, EvaluationMemoryLimit) as error:
                worker.kill()
                worker.communicate()
                rows.append({'id': case['id'], 'status': 'NOT_EVALUATED_TIMEOUT'
                    if isinstance(error, subprocess.TimeoutExpired) else 'NOT_EVALUATED_RESOURCE_LIMIT'})
            continue
        root = Path('.orchestrator/external-v14/snapshots') / case['id']
        before, inventory = python_source_hash(root), snapshot_inventory(root)
        if before != case['source_python_sha256']:
            raise ValueError('Snapshot divergente.')
        policy = PathPolicy((root.resolve(),))
        snapshot = ProjectIndexBuilderV2(root, path_policy=policy).build()
        problem = case['problem_formulations'][0]
        context = build_problem_context(problem, snapshot, ProjectCandidateGenerator().generate(problem, snapshot), policy)
        result = behavioral_interventions(context, snapshot, policy, case['oracle'])
        blob = json.dumps(result, sort_keys=True).encode()
        sha = hashlib.sha256(blob).hexdigest()
        path = archive / (sha + '.json.gz')
        path.write_bytes(gzip.compress(blob, mtime=0))
        if before != python_source_hash(root) or inventory != snapshot_inventory(root):
            raise AssertionError('Snapshot modificado.')
        rows.append({'id': case['id'], 'status': 'COMPLETED', 'experiments': result['experiments'],
                     'raw_trace_sha256': sha, 'raw_trace_archive': str(path)})
    if args.case_id and not rows:
        raise ValueError('ID fora de development selado.')
    args.output.write_text(json.dumps({'version': 14, 'experimental_only': True, 'results': rows,
        'qwen_calls': 0, 'limitation': 'C path oracle not fully adjudicated; E selector not measured offline.'}, indent=2) + '\n')


if __name__ == '__main__':
    main()
