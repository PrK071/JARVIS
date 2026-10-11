"""Estudo opt-in sem preflight duplicado nem execucao de projeto externo."""
import argparse
import gzip
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path


def evidence_information_gain(before, after):
    """Mudancas observaveis, nao probabilidade nem prova de melhoria causal."""
    fields = ('api_ids', 'relation_ids', 'effect_ids', 'contract_ids', 'root_ids', 'supported_root_ids')
    added = {key: sorted(set(after.get(key, ())) - set(before.get(key, ()))) for key in fields}
    return {'added': added, 'state_before': before.get('state'), 'state_after': after.get('state'),
        'changed': any(added.values()) or before.get('state') != after.get('state'),
        'causal_correctness_improved': after.get('correct_candidate_exists', False)
            and not before.get('correct_candidate_exists', False),
        'limitation': 'Novas entidades nao demonstram causalidade; confirmar com adjudicacao independente.'}


def runtime_hashes(runtime):
    return {p.relative_to(runtime).as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in sorted((runtime / 'tern/orchestrator/predictive').glob('*.py'))}


def run_worker(args):
    # Somente o runtime confiavel JARVIS; nunca adicionar snapshots ao sys.path.
    sys.path.insert(0, str(args.runtime.resolve()))
    from tern.orchestrator.client import LlamaClient
    from tern.orchestrator.predictive import service as runtime_service
    from tern.orchestrator.predictive.evaluation import CountingReasoner
    from tern.orchestrator.predictive.external_evaluation import (
        load_external_cases, matches_origin, python_source_hash, snapshot_inventory)
    from tern.orchestrator.predictive.operational import observe_checkpoints, observe_phase
    from tern.orchestrator.security import PathPolicy
    from scripts.predictive_checkpoints import CheckpointJournal

    case = next(c for c in load_external_cases(args.corpus, split=args.split) if c['id'] == args.case)
    input_ = case['inputs'][args.input_mode]
    if not input_['available']:
        raise ValueError('Modo indisponivel nao pode ser avaliado.')
    root = (args.snapshots / case['id']).resolve(strict=True)
    if not root.is_relative_to(args.snapshots.resolve()) or args.archive.resolve().is_relative_to(root):
        raise ValueError('Raiz ou artefatos invalidos.')
    before = python_source_hash(root), snapshot_inventory(root)
    if before[0] != case['source_python_sha256']:
        raise ValueError('Integridade do snapshot invalida.')
    directory = args.archive
    if not args.output.resolve().is_relative_to(directory.resolve()) or directory.is_symlink():
        raise ValueError('Resultado do worker deve permanecer em artefatos autorizados.')
    directory.mkdir(parents=True, exist_ok=True)
    journal = CheckpointJournal(directory / 'checkpoints.jsonl', args.snapshots)
    captured = {}
    original = runtime_service.build_contract_contexts

    def observe_context(context, policy):
        # Observador no harness isolado: nao insere nem altera evidencia.
        captured['context'] = context
        result = original(context, policy)
        captured['contracts'] = result
        return result

    runtime_service.build_contract_contexts = observe_context
    class Reasoner:
        def __init__(self):
            self.client = LlamaClient('http://127.0.0.1:8080', 180)
        def chat(self, *messages, **kwargs):
            result = observe_phase('qwen_request', self.client.chat, *messages, **kwargs)
            # Preservar resposta antes da validacao: timeout pos-modelo nao a apaga.
            with gzip.open(directory / 'model-response.json.gz', 'wt', encoding='utf-8') as f:
                json.dump({'messages': messages, 'response_format': kwargs.get('response_format'),
                    'response': result}, f, ensure_ascii=False)
            return result
    reasoner = CountingReasoner(Reasoner())
    started = time.perf_counter()
    with observe_checkpoints(journal):
        report = runtime_service.PredictiveDecisionService(reasoner, path_policy=PathPolicy((root,))).predict(input_['text'], root)
    if before != (python_source_hash(root), snapshot_inventory(root)):
        raise AssertionError('Snapshot externo alterado.')
    context = captured.get('context')
    contracts = captured.get('contracts', ())
    raw = report.as_dict()
    roots = raw['root_cause_candidates']
    correct_ids = {r['id'] for r in roots if matches_origin(r, case['oracle'])}
    selected_ids = {h['root_cause_id'] for h in raw.get('hypotheses', ())}
    candidates = raw.get('candidates', ())
    winner = next((c for c in candidates if c['id'] == raw.get('recommended_candidate_id')), None)
    support = raw.get('evidence_sufficiency') or {}
    graph = context.behavioral_graph if context else None
    location_correct = lambda h: (h['path'] == case['oracle']['root_path'] and h['symbol'] == case['oracle']['root_symbol'])
    investigation = raw.get('investigable_hypotheses', ())
    safe = {'filesystem_mutations': 0, 'tool_dispatches': reasoner.tool_dispatches,
        'authority_grants': 0, 'execution_authorized': int(report.execution_authorized),
        'destructive_actions': 0, 'forbidden_recommendations': int(bool(winner and not winner['eligible']))}
    checkpoint_rows = [json.loads(line) for line in (directory / 'checkpoints.jsonl').read_text().splitlines()]
    states = {'api_ids': list(graph.entry_ids) if graph else [],
        'relation_ids': [r.id for r in graph.relations] if graph else [],
        'effect_ids': [e.id for e in graph.effects] if graph else [],
        'contract_ids': [c.id for c in contracts if c.contract_exists],
        'root_ids': [r['id'] for r in roots], 'supported_root_ids': support.get('supported_root_ids', []),
        'state': support.get('classification', 'BASELINE_WITHOUT_TRIAGE'),
        'correct_candidate_exists': bool(correct_ids)}
    payload = {'input_source': input_['source'], 'input_text': input_['text'], 'decision': raw,
        'ledger': context.evidence_ledger.as_dict() if context else None, 'states': states}
    with gzip.open(directory / 'trace.json.gz', 'wt', encoding='utf-8') as f:
        json.dump(payload, f, ensure_ascii=False)
    ledger_ids = {a.id for a in context.evidence_ledger.atoms} if context else set()
    refs = [i for r in roots for i in r['evidence_ids']] + [i for h in investigation for i in h['evidence_ids']]
    row = {'case_id': case['id'], 'input_mode': args.input_mode, 'status': 'COMPLETED',
        'wall_seconds': time.perf_counter()-started, 'phases': dict(report.phase_seconds),
        'qwen_requests': reasoner.requests, 'qwen_request_ms': reasoner.request_ms,
        'peak_working_set': max((c.get('working_set_bytes') or 0 for c in checkpoint_rows), default=0),
        'state': states, 'root_states': support.get('roots', []),
        'requests': raw.get('diagnostic_evidence_requests', []),
        'correct_file_retrieved': bool(context and case['oracle']['root_path'] in context.related_files),
        'correct_candidate_exists': bool(correct_ids),
        'root_correct': bool(selected_ids) and selected_ids.issubset(correct_ids),
        'repair_correct': bool(winner and winner['root_cause_id'] in correct_ids
            and winner['strategy_kind'] in case['oracle']['repair_strategies']
            and all(t['path'] in case['oracle']['target_paths'] for t in winner.get('repair_targets', ()))),
        'recommended': winner is not None, 'hypotheses_emitted': len(investigation),
        'correct_hypothesis_locations': sum(location_correct(h) for h in investigation),
        'hypothesis_location_population': len(investigation),
        'evidence_references_valid': sum(i in ledger_ids for i in refs),
        'evidence_reference_population': len(refs), 'safety': safe,
        'original_reveals_location': case['original_reveals_location'],
        'trace_sha256': hashlib.sha256((directory / 'trace.json.gz').read_bytes()).hexdigest(),
        'last_checkpoint': checkpoint_rows[-1] if checkpoint_rows else None}
    if report.failure_reason and report.failure_reason.value == 'REASONER_UNAVAILABLE':
        row['status'] = 'NOT_EVALUATED_QWEN_FAILURE'
        row['excluded_from_causal_metrics'] = True
        row['qwen_error_checkpoints'] = [c for c in checkpoint_rows
            if c['phase'] == 'qwen_request' and c['state'] == 'ERROR']
    args.output.write_text(json.dumps(row, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def summarize(rows, all_cases):
    from tern.orchestrator.predictive.external_evaluation import rate
    result = {}
    for mode in 'ABCD':
        eligible = [(c['id'], mode) for c in all_cases if c['inputs'][mode]['available']]
        group = [r for r in rows if r['input_mode'] == mode]
        completed = [r for r in group if r['status'] == 'COMPLETED']
        recommended = [r for r in completed if r['recommended']]
        n = len(completed)
        metrics = {'completion_rate': rate(n, len(eligible), 'presealed independent defects eligible for '+mode)}
        for key in ('correct_file_retrieved', 'correct_candidate_exists', 'root_correct', 'repair_correct'):
            metrics[key] = rate(sum(r[key] for r in completed), n, 'completed independent defects, mode '+mode)
        metrics['recommendation_precision'] = rate(sum(r['repair_correct'] for r in recommended), len(recommended), 'recommendations')
        metrics['false_abstention_rate'] = rate(sum(not r['recommended'] for r in completed), n, 'completed positive defects')
        metrics['investigable_hypothesis_coverage'] = rate(sum(bool(r['hypotheses_emitted']) for r in completed), n, 'completed defects')
        metrics['hypothesis_location_precision'] = rate(sum(r['correct_hypothesis_locations'] for r in completed),
            sum(r['hypothesis_location_population'] for r in completed), 'hypotheses; source-location proxy not causal confirmation')
        metrics['evidence_reference_validity'] = rate(sum(r['evidence_references_valid'] for r in completed),
            sum(r['evidence_reference_population'] for r in completed), 'candidate and investigation evidence IDs')
        metrics['required_evidence_precision'] = rate(0, 0, 'undefined: no independent adjudication of acquisition efficacy')
        walls = sorted(r.get('wall_seconds', 0) for r in group)
        result[mode] = {'eligible_defects': len(eligible), 'completed_defects': n,
            'metrics': metrics, 'qwen_requests': sum(r.get('qwen_requests', 0) for r in group),
            'median_wall_seconds': statistics.median(walls) if walls else None,
            'p95_wall_seconds': walls[max(0, (95*len(walls)+99)//100-1)] if walls else None,
            'states': {state: sum(r['state']['state'] == state for r in completed)
                for state in sorted({r['state']['state'] for r in completed})}}
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('development', 'holdout'), required=True)
    parser.add_argument('--runtime', type=Path, default=Path.cwd())
    parser.add_argument('--corpus', type=Path, default=Path('tests/data/predictive/external_v16'))
    parser.add_argument('--snapshots', type=Path, default=Path('D:/JARVIS-evaluation-v16/external-v16/snapshots'))
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--case')
    parser.add_argument('--input-mode', choices=tuple('ABCD'))
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(args.snapshots.resolve()) or args.archive.resolve().is_relative_to(args.snapshots.resolve()):
        raise ValueError('Estudo nao pode gravar em projetos externos.')
    if args.output.exists():
        raise ValueError('Nao sobrescrever experimento.')
    if args.case:
        if not args.live or not args.input_mode:
            raise ValueError('Worker live requer modo de input.')
        run_worker(args)
        return
    if not args.live:
        raise ValueError('Live explicitamente opt-in; nao executar projetos externos.')
    from tern.orchestrator.predictive.external_evaluation import load_external_cases
    from scripts.predictive_external_eval import _communicate_bounded, EvaluationMemoryLimit
    from scripts.predictive_checkpoints import last_checkpoint
    cases = load_external_cases(args.corpus, split=args.split)
    manifest = json.loads((args.corpus / 'manifest.json').read_text())
    source_before = runtime_hashes(args.runtime.resolve())
    if args.split == 'holdout':
        marker = args.archive / 'holdout-opened.json'
        marker.parent.mkdir(parents=True, exist_ok=True)
        with marker.open('x', encoding='utf-8') as f:
            json.dump({'corpus': manifest['cases_sha256'], 'runtime': source_before}, f)
    rows = []
    for case in cases:
        for mode, variant in case['inputs'].items():
            if not variant['available']:
                continue
            directory = args.archive / case['id'] / mode
            directory.mkdir(parents=True, exist_ok=True)
            output = directory / 'result.json'
            if output.exists() or (directory / 'checkpoints.jsonl').exists():
                raise ValueError('Tentativa ja existe; nao repetir holdout.')
            command = [sys.executable, '-m', 'scripts.predictive_v16_evidence_study', '--split', args.split,
                '--runtime', str(args.runtime.resolve()), '--case', case['id'], '--input-mode', mode,
                '--corpus', str(args.corpus.resolve()), '--snapshots', str(args.snapshots.resolve()),
                '--archive', str(directory.resolve()), '--output', str(output.resolve()), '--live']
            environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1')
            started = time.monotonic()
            worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, encoding='utf-8', errors='replace', env=environment)
            try:
                _, stderr = _communicate_bounded(worker, command, manifest['budget_seconds'], manifest['memory_mb'])
                if worker.returncode or not output.exists():
                    raise RuntimeError(stderr[-1800:])
                row = json.loads(output.read_text(encoding='utf-8'))
            except (subprocess.TimeoutExpired, EvaluationMemoryLimit, RuntimeError) as error:
                if worker.poll() is None:
                    worker.kill()
                worker.communicate()
                checkpoint = last_checkpoint(directory / 'checkpoints.jsonl')
                row = {'case_id': case['id'], 'input_mode': mode,
                    'status': 'NOT_EVALUATED_TIMEOUT' if isinstance(error, subprocess.TimeoutExpired)
                        or 'NOT_EVALUATED_TIMEOUT' in str(error) else 'NOT_EVALUATED_INFRASTRUCTURE',
                    'diagnostic': str(error)[-1800:], 'last_checkpoint': checkpoint,
                    'timeout_phase': checkpoint['phase'] if checkpoint else None,
                    'wall_seconds': time.monotonic()-started, 'excluded_from_causal_metrics': True}
                if isinstance(error, EvaluationMemoryLimit):
                    row['status'] = 'NOT_EVALUATED_RESOURCE_LIMIT'
                journal_path = directory / 'checkpoints.jsonl'
                events = [json.loads(line) for line in journal_path.read_text().splitlines()] if journal_path.exists() else []
                row['qwen_requests'] = sum(e['phase'] == 'qwen_request' and e['state'] == 'START' for e in events)
                row['qwen_request_ms'] = [e['elapsed_seconds']*1000 for e in events
                    if e['phase'] == 'qwen_request' and e['state'] == 'END']
                row['peak_working_set'] = max((e.get('working_set_bytes') or 0 for e in events), default=0)
            rows.append(row)
            print(json.dumps({'case_id': case['id'], 'input_mode': mode, 'status': row['status']}), flush=True)
    if source_before != runtime_hashes(args.runtime.resolve()):
        raise ValueError('Runtime mudou durante estudo.')
    paired = []
    for case in cases:
        base = next((r for r in rows if r['case_id'] == case['id'] and r['input_mode'] == 'A' and r['status'] == 'COMPLETED'), None)
        if base:
            for row in rows:
                if row['case_id'] == case['id'] and row['input_mode'] != 'A' and row['status'] == 'COMPLETED':
                    paired.append({'case_id': case['id'], 'mode': row['input_mode'],
                        'gain': evidence_information_gain(base['state'], row['state'])})
    value = {'version': 16, 'corpus_sha256': manifest['cases_sha256'], 'runtime_sources': source_before,
        'split': args.split, 'summary': summarize(rows, cases), 'results': rows, 'information_gain': paired,
        'limitations': ['D unavailable: no new pre-fix runtime evidence; existing original logs already in A.',
            'C available only when original report cited preexisting base-commit test; no gold tests inserted.',
            'Independent defects and input formulations are distinct populations.',
            'Baseline external qualification invalid after post-v15-seal correction; diagnostic comparison only.']}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


if __name__ == '__main__':
    main()
