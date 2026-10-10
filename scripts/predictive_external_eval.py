"""Explicit opt-in live external evaluation; no external project execution."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tern.orchestrator.client import LlamaClient
from tern.orchestrator.predictive.external_evaluation import (
    evaluate_external_cases, load_external_cases, summarize_external_results,
)


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
                   '--corpus', str(args.corpus), '--snapshots', str(args.snapshots)]
        worker = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  text=True, encoding='utf-8', errors='replace')
        try:
            stdout, stderr = worker.communicate(timeout=args.case_timeout)
        except subprocess.TimeoutExpired:
            worker.kill()
            worker.communicate()
            incomplete.append({'id': case['id'], 'status': 'NOT_COMPLETED_OPERATIONAL_LIMIT',
                               'budget_seconds': args.case_timeout,
                               'excluded_from_quality_metrics': True})
            print(json.dumps(incomplete[-1]), flush=True)
            continue
        if worker.returncode or not output.exists():
            incomplete.append({'id': case['id'], 'status': 'EVALUATOR_INFRASTRUCTURE_ERROR',
                               'exit_code': worker.returncode, 'diagnostic': stderr[-1600:],
                               'excluded_from_quality_metrics': True})
            print(json.dumps(incomplete[-1]), flush=True)
            continue
        rows.extend(json.loads(output.read_text(encoding='utf-8'))['results'])
        print(json.dumps({'id': case['id'], 'status': 'COMPLETED'}), flush=True)
    from tern.orchestrator.predictive.coverage_evaluation import summarize_layer_coverage
    summary = summarize_external_results(rows)
    summary['layer_coverage'] = summarize_layer_coverage(rows)
    summary.update(n_total_defects=len(cases), n_evaluable_defects=len(cases)-len(incomplete),
                   n_incomplete_defects=len(incomplete))
    return {'schema_version': 13, 'summary': summary, 'results': rows,
            'operational_failures': incomplete, 'case_budget_seconds': args.case_timeout,
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
    parser.add_argument("--case-id", help="Harness-only independent defect filter")
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
    source_hashes = {
        str(path): hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(Path("tern/orchestrator/predictive").glob("*.py"))
        if path.name != "external_evaluation.py"
    }
    value = evaluate_with_budget(args, cases) if args.case_timeout else evaluate_external_cases(cases, args.snapshots,
        args.archive / args.output.stem,
        reasoner=LlamaClient("http://127.0.0.1:8080", 180) if args.mode == "live" else None)
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
