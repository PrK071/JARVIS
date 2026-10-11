"""Freeze, integridade, limitacoes e gates medidos separadamente."""
import argparse
import gzip
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from scripts.predictive_v16_evidence_study import runtime_hashes
from tern.orchestrator.predictive.external_evaluation import rate

RESULTS = Path('tests/data/predictive/results')


def hashes():
    result = runtime_hashes(Path.cwd())
    paths = [Path('tern/orchestrator/project_intelligence_v2.py'), Path('tern/orchestrator/cli.py'),
        *Path('scripts').glob('*v16*.py'), Path('scripts/predictive_checkpoints.py'),
        Path('scripts/predictive_external_eval.py')]
    result.update({p.as_posix(): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in paths})
    return result


def journal_audit(archive):
    rows = []
    for path in sorted(archive.rglob('checkpoints.jsonl')):
        events = [json.loads(line) for line in path.read_text().splitlines()]
        starts = [e for e in events if e['phase'] == 'qwen_request' and e['state'] == 'START']
        ends = [e for e in events if e['phase'] == 'qwen_request' and e['state'] in ('END', 'ERROR')]
        analysis = [e for e in events if e['phase'] == 'qwen_analysis' and e['state'] == 'START']
        rows.append({'journal': path.relative_to(archive).as_posix(), 'requests_started': len(starts),
            'requests_finished': len(ends), 'pre_model_analysis_seconds': starts[0]['wall_seconds']-analysis[0]['wall_seconds'] if starts and analysis else None,
            'model_inclusive_queue_seconds': [e['elapsed_seconds'] for e in ends],
            'last_checkpoint': events[-1] if events else None,
            'peak_sampled_working_set': max((e.get('working_set_bytes') or 0 for e in events), default=0)})
    return {'rows': rows, 'requests_started': sum(r['requests_started'] for r in rows),
        'requests_finished': sum(r['requests_finished'] for r in rows),
        'limitation': 'Queue/generation inseparaveis; working set amostrado nao e limite rigido de alocacao.'}


def preservation(archive):
    rows = []
    for path in sorted(archive.rglob('*')):
        if path.is_file():
            if path.is_symlink() or not path.resolve().is_relative_to(archive.resolve()):
                raise ValueError('Artefato com escape.')
            raw = path.read_bytes()
            row = {'path': path.relative_to(archive).as_posix(), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            if path.suffix == '.gz':
                row['uncompressed_sha256'] = hashlib.sha256(gzip.decompress(raw)).hexdigest()
            rows.append(row)
    historical = []
    for report in ('v13-artifact-preservation.json', 'v15-v13-artifacts-reverified.json',
            'v15-previous-artifacts-reverified.json', 'v15-artifact-preservation-final.json'):
        manifest = json.loads((RESULTS / report).read_text())
        directory = Path(manifest.get('destination', manifest.get('local_archive')))
        checked = 0
        for item in manifest.get('traces', []):
            if 'relative_path' not in item:
                continue
            path = (directory / item['relative_path']).resolve(strict=True)
            if not path.is_relative_to(directory.resolve()):
                raise ValueError('Trace historico fora do backup.')
            if hashlib.sha256(path.read_bytes()).hexdigest() != item['compressed_sha256']:
                raise ValueError('Integridade historica invalida.')
            checked += 1
        if manifest.get('portable_zip_path'):
            path = Path(manifest['portable_zip_path'])
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['portable_zip_sha256']:
                raise ValueError('ZIP historico invalido.')
        historical.append({'report': report, 'destination': str(directory), 'traces_verified': checked,
            'portable_zip_verified': bool(manifest.get('portable_zip_path'))})
    return {'archive': str(archive), 'inventory': rows, 'historical': historical,
        'temporary_sources_deleted': False, 'risk': 'Copias locais no mesmo host; backup off-host ainda recomendado.'}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--freeze', action='store_true')
    parser.add_argument('--full-tests-passed', action='store_true')
    parser.add_argument('--archive', type=Path, default=Path('D:/JARVIS-evaluation-v16/artifacts/v16'))
    args = parser.parse_args()
    freeze = RESULTS / 'v16-runtime-freeze.json'
    corpus = json.loads(Path('tests/data/predictive/external_v16/manifest.json').read_text())
    if args.freeze:
        with freeze.open('x', encoding='utf-8') as f:
            json.dump({'frozen_at': datetime.now(timezone.utc).isoformat(), 'corpus_seal': corpus['cases_sha256'],
                'sources': hashes(), 'holdout_thresholds': corpus['holdout_thresholds'],
                'coverage_candidate_recall_threshold': .75, 'investigation_coverage_threshold': .75,
                'required_evidence_precision_threshold': .8, 'budget_unit': '240s per independent worker/input',
                'undefined_metrics_never_pass': True,
                'small_purposive_population_never_establishes_broad_production_readiness': True}, f, indent=2)
        return
    seal = json.loads(freeze.read_text())
    if seal['sources'] != hashes() or seal['corpus_seal'] != corpus['cases_sha256']:
        raise ValueError('Runtime/evaluator mudaram apos freeze; nao promover.')
    hold_path = RESULTS / 'v16-external-holdout-one-shot.json'
    hold = json.loads(hold_path.read_text())
    dev = json.loads((RESULTS / 'v16-development-after.json').read_text())
    complete = all(s['completed_defects'] == s['eligible_defects'] for study in (dev, hold)
        for s in study['summary'].values() if s['eligible_defects'])
    rows = [r for r in hold['results'] if r['status'] == 'COMPLETED']
    metrics = hold['summary']['A']['metrics']
    def meets(key, threshold):
        value = metrics[key]['rate']
        return value is not None and value >= threshold
    refs = sum(r['evidence_reference_population'] for r in rows)
    valid = sum(r['evidence_references_valid'] for r in rows)
    synthetic = json.loads((RESULTS / 'v16-synthetic-structural.json').read_text())
    structural = all(m['value'] == 1 for m in synthetic['metrics'].values())
    before = json.loads((RESULTS / 'v16-static-profile-before.json').read_text())
    after = json.loads((RESULTS / 'v16-static-profile-after.json').read_text())
    exact = before['semantic_sha256'] == after['semantic_sha256']
    safety = {k: sum(r['safety'][k] for r in rows) for k in rows[0]['safety']} if rows else {}
    grounded = bool(refs) and valid == refs and not any(safety.values())
    evidence_precision = metrics['required_evidence_precision']['rate']
    gates = {'PREDICTIVE_SYNTHETIC_READY': structural and exact and args.full_tests_passed,
        'DIAGNOSTIC_TRIAGE_READY': complete and evidence_precision is not None
            and evidence_precision >= seal['required_evidence_precision_threshold'] and grounded,
        'OPERATIONAL_EVALUATION_READY': complete,
        'INVESTIGATION_CAPABILITY_READY': complete and grounded
            and meets('investigable_hypothesis_coverage', seal['investigation_coverage_threshold'])
            and meets('hypothesis_location_precision', seal['holdout_thresholds']['hypothesis_location_precision']),
        'PREDICTIVE_EXTERNAL_COVERAGE_READY': complete and grounded
            and meets('correct_candidate_exists', seal['coverage_candidate_recall_threshold']),
        'PREDICTIVE_EXTERNAL_DECISION_READY': complete and grounded
            and meets('root_correct', seal['holdout_thresholds']['root_validity'])
            and meets('recommendation_precision', seal['holdout_thresholds']['root_validity']),
        'SIMULATION_RUNTIME_READY': False}
    support = Counter(r['state'] for row in rows for r in row['root_states'])
    value = {'version': 16, 'gates': gates, 'corpus_sha256': corpus['cases_sha256'],
        'runtime_freeze': freeze.as_posix(), 'full_tests_passed': args.full_tests_passed,
        'holdout_completion': rate(len(rows), len(hold['results']), 'presealed eligible inputs; not independent defects'),
        'root_support_funnel': dict(support), 'primary_independent_defect_metrics': metrics,
        'safety': safety, 'evidence_reference_validity': rate(valid, refs, 'candidate/investigation IDs'),
        'deterministic_profile_semantics_equal': exact,
        'stop_condition': 'STATIC_EVIDENCE_INSUFFICIENT_FOR_RELIABLE_DIAGNOSIS',
        'stop_scope': 'Motor e informacao estudados; nao demonstra impossibilidade teorica de analise estatica.',
        'next_milestone': 'Secure Sandbox Validation -> Controlled Runtime Evidence -> Causal Reassessment',
        'limitations': ['Localizacao de hypothesis nao confirma causa; candidato correto e proxy de localizacao.',
            'required_evidence_precision undefined: necessidade plausivel ainda sem eficacia externa adjudicada.',
            'B pode materializar expectativa UNKNOWN como metadado relatado; mudanca de ID nao e ganho causal.',
            'D indisponivel; um unico C nao permite medir valor geral de observacoes runtime.',
            'Sintetico: regressao estrutural+unitaria, nao repeticao de todo holdout live v10.']}
    for name, data in (('v16-stage-gate.json', value), ('v16-artifact-preservation.json', preservation(args.archive)),
            ('v16-operational-journals.json', journal_audit(args.archive))):
        with (RESULTS / name).open('x', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.write('\n')


if __name__ == '__main__':
    main()
