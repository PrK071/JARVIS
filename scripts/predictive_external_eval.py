"""Explicit opt-in live external evaluation; no external project execution."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from tern.orchestrator.client import LlamaClient
from tern.orchestrator.predictive.external_evaluation import (
    evaluate_external_cases, load_external_cases, summarize_external_results,
)
from tern.orchestrator.predictive.operational import observe_checkpoints, observe_phase
from .predictive_checkpoints import CheckpointJournal, last_checkpoint


def _working_set_bytes(worker):
    """Memoria do nosso worker Windows, nao de processos arbitrarios."""
    if sys.platform != 'win32' or not hasattr(worker, '_handle'):
        return None
    import ctypes
    from ctypes import wintypes

    class MemoryCounters(ctypes.Structure):
        _fields_ = [('cb', wintypes.DWORD), ('faults', wintypes.DWORD)] + [
            (name, ctypes.c_size_t) for name in ('peak', 'working', 'quota_peak_paged',
                'quota_paged', 'quota_peak_nonpaged', 'quota_nonpaged', 'pagefile', 'peak_pagefile')]

    counters = MemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    function = ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes = [wintypes.HANDLE, ctypes.POINTER(MemoryCounters), wintypes.DWORD]
    function.restype = wintypes.BOOL
    if function(int(worker._handle), ctypes.byref(counters), counters.cb):
        return int(counters.working)
    return None


class EvaluationMemoryLimit(RuntimeError):
    pass


def _communicate_bounded(worker, command, timeout, memory_mb):
    if not memory_mb:
        return worker.communicate(timeout=timeout)
    deadline = time.monotonic() + timeout
    while True:
        memory = _working_set_bytes(worker)
        if memory is not None and memory > memory_mb * 1024 * 1024:
            raise EvaluationMemoryLimit('NOT_EVALUATED_RESOURCE_LIMIT')
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, timeout)
        try:
            return worker.communicate(timeout=min(1, remaining))
        except subprocess.TimeoutExpired:
            continue


def evaluate_with_budget(args, cases):
    """Only our read-only evaluator runs on the host, never project code."""
    rows, incomplete = [], []
    for case in cases:
        directory = args.archive / args.output.stem / case['id']
        directory.mkdir(parents=True, exist_ok=True)
        output = directory / 'result.json'
        if output.exists():
            raise ValueError('Resultado de caso ja existe; nao reutilizar holdout silenciosamente.')
        command = [sys.executable, '-m', 'scripts.predictive_external_eval',
                   '--split', args.split, '--mode', args.mode, '--case-id', case['id'],
                   '--output', str(output), '--archive', str(directory),
                   '--corpus', str(args.corpus), '--snapshots', str(args.snapshots),
                   '--checkpoints', str(directory / 'checkpoints.jsonl')]
        worker_started = time.monotonic()
        worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding='utf-8', errors='replace')
        try:
            stdout, stderr = _communicate_bounded(worker, command, args.case_timeout,
                getattr(args, 'case_memory_mb', 0))
        except (subprocess.TimeoutExpired, EvaluationMemoryLimit) as error:
            worker.kill()
            worker.communicate()
            incomplete.append({'id': case['id'], 'status': (
                               'NOT_EVALUATED_RESOURCE_LIMIT' if isinstance(error, EvaluationMemoryLimit)
                               else 'NOT_EVALUATED_TIMEOUT'),
                               'budget_seconds': args.case_timeout,
                               'last_checkpoint': last_checkpoint(directory / 'checkpoints.jsonl'),
                               'excluded_from_quality_metrics': True})
            print(json.dumps(incomplete[-1]), flush=True)
            continue
        if worker.returncode or not output.exists():
            incomplete.append({'id': case['id'], 'status': (
                               'NOT_EVALUATED_TIMEOUT' if 'NOT_EVALUATED_TIMEOUT' in stderr
                               else 'EVALUATOR_INFRASTRUCTURE_ERROR'),
                               'exit_code': worker.returncode, 'diagnostic': stderr[-1600:],
                               'last_checkpoint': last_checkpoint(directory / 'checkpoints.jsonl'),
                               'excluded_from_quality_metrics': True})
            print(json.dumps(incomplete[-1]), flush=True)
            continue
        completed = json.loads(output.read_text(encoding='utf-8'))['results']
        for row in completed:
            row['total_worker_seconds'] = time.monotonic()-worker_started
        rows.extend(completed)
        print(json.dumps({'id': case['id'], 'status': 'COMPLETED'}), flush=True)
    from tern.orchestrator.predictive.coverage_evaluation import summarize_layer_coverage
    summary = summarize_external_results(rows)
    summary['layer_coverage'] = summarize_layer_coverage(rows)
    from tern.orchestrator.predictive.investigation_evaluation import investigation_metrics
    summary['investigation'] = investigation_metrics(rows, incomplete)
    summary.update(n_total_defects=len(cases), n_evaluable_defects=len(cases)-len(incomplete),
                   n_incomplete_defects=len(incomplete))
    return {'schema_version': 14, 'summary': summary, 'results': rows,
            'operational_failures': incomplete, 'case_budget_seconds': args.case_timeout,
            'case_memory_mb': getattr(args, 'case_memory_mb', 0),
            'memory_monitor': 'Windows working-set polling, one-second intervals; not a hard allocation cap',
            'population_note': 'Incomplete cases excluded explicitly, not counted as engine errors. Readiness requires complete coverage.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("development", "holdout", "all"), required=True)
    parser.add_argument("--mode", choices=("structural", "live"), default="structural")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("tests/data/predictive/external_v12"))
    parser.add_argument("--snapshots", type=Path, default=Path(".orchestrator/external-v12/snapshots"))
    parser.add_argument("--archive", type=Path, default=Path(".orchestrator/predictive-artifacts/v12"))
    parser.add_argument("--case-timeout", type=float, default=0)
    parser.add_argument("--case-memory-mb", type=int, default=1024)
    parser.add_argument("--case-id", help="Harness-only independent defect filter")
    parser.add_argument('--checkpoints', type=Path, help='Journal do worker fora dos snapshots')
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Nao sobrescrever resultado historico.")
    corpus = args.corpus
    cases = load_external_cases(corpus, split=args.split)
    if args.case_id:
        cases = tuple(case for case in cases if case['id'] == args.case_id)
        if len(cases) != 1:
            raise ValueError('Caso nao pertence ao split selado.')
    if args.case_timeout < 0:
        raise ValueError('Timeout de caso invalido.')
    if args.case_memory_mb < 0:
        raise ValueError('Limite de memoria invalido.')
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(Path("tern/orchestrator/predictive").glob("*.py"))
        if path.name != "external_evaluation.py"
    }
    journal = CheckpointJournal(args.checkpoints, args.snapshots) if args.checkpoints else None
    class CheckpointReasoner:
        def __init__(self):
            self.client = LlamaClient('http://127.0.0.1:8080', 180)
        def chat(self, *messages, **kwargs):
            return observe_phase('qwen_request', self.client.chat, *messages, **kwargs)
    with observe_checkpoints(journal):
        value = evaluate_with_budget(args, cases) if args.case_timeout else evaluate_external_cases(cases, args.snapshots,
            args.archive / args.output.stem, reasoner=CheckpointReasoner() if args.mode == 'live' else None)
    value["engine_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    value["corpus_seal"] = json.loads((corpus/"manifest.json").read_text())["cases_sha256"]
    value["split"] = args.split
    if any(hashlib.sha256(Path(path).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != digest
           for path, digest in source_hashes.items()):
        raise ValueError('Runtime mudou durante avaliacao; resultado nao promovido.')
    value["runtime_source_sha256"] = source_hashes
    args.output.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value["summary"], indent=2))


if __name__ == "__main__":
    main()
