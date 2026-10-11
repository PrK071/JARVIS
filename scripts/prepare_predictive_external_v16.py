"""Inventario de relatos: dados externos, sem execucao de projetos."""
import argparse
import ast
import hashlib
import io
import json
import re
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PurePosixPath
from .prepare_predictive_external_v12 import projection_path, source_hash


ADJUDICATION = {
    'psf__requests-1142': ('development', 'requests/models.py', 'PreparedRequest.prepare_content_length', 'GET sem body recebe header Content-Length indevido.'),
    'psf__requests-2931': ('development', 'requests/models.py', 'RequestEncodingMixin._encode_params', 'Conversao de payload binario para texto viola preservacao do body.'),
    'pytest-dev__pytest-10051': ('development', 'src/_pytest/logging.py', 'LogCaptureHandler.reset', 'Substituicao da lista deixa o consumidor com referencia antiga.'),
    'sphinx-doc__sphinx-9367': ('development', 'sphinx/pycode/ast.py', '_UnparseVisitor.visit_Tuple', 'Representacao de tuple unitario perde virgula semantica.'),
    'psf__requests-1724': ('holdout', 'requests/sessions.py', 'Session.request', 'Metodo unicode propaga tipo incompatível com a camada de transporte Python2.'),
    'psf__requests-1921': ('holdout', 'requests/sessions.py', 'merge_setting', 'Valores None do session setting nao sao removidos do merge.'),
    'psf__requests-5414': ('holdout', 'requests/models.py', 'PreparedRequest.prepare_url', 'Label vazio nao e rejeitado na validacao local da URL.'),
    'psf__requests-6028': ('holdout', 'requests/utils.py', 'prepend_scheme_if_needed', 'Reconstrucao da URL omite autenticacao do proxy.'),
    'pytest-dev__pytest-5262': ('holdout', 'src/_pytest/capture.py', 'EncodedFile', 'Mode herdado do buffer anuncia binary apesar do contrato de write textual.'),
    'pytest-dev__pytest-10081': ('holdout', 'src/_pytest/unittest.py', 'TestCaseFunction.runtest', 'Skip de classe nao participa do guard de teardown em usepdb.'),
    'sphinx-doc__sphinx-8621': ('holdout', 'sphinx/builders/html/transforms.py', 'KeyboardTransform', 'Regex de separadores trata keystrokes isolados como separadores.'),
    'sphinx-doc__sphinx-9711': ('holdout', 'sphinx/extension.py', 'verify_needs_extensions', 'Comparacao lexical de versoes contradiz ordenacao de versoes.'),
}


def seal(rows, historical):
    from .predictive_v16_study import input_variants
    target = Path('tests/data/predictive/external_v16')
    if (target / 'manifest.json').exists() or historical.intersection(ADJUDICATION):
        raise ValueError('Corpus ja selado ou caso historico.')
    cache = Path('D:/JARVIS-evaluation-v16/external-v16')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    lookup = {r['instance_id']: r for r in rows}
    cases = []
    for case_id, (split, path, symbol, cause) in ADJUDICATION.items():
        row = lookup[case_id]
        url = f"https://codeload.github.com/{row['repo']}/tar.gz/{row['base_commit']}"
        archive = cache / f'{case_id}.tar.gz'
        if not archive.exists():
            with opener.open(url, timeout=60) as response:
                raw = response.read(64_000_001)
            if len(raw) > 64_000_000:
                raise ValueError('Archive excede limite.')
            archive.write_bytes(raw)
        root = cache / 'snapshots' / case_id
        total = 0
        with tarfile.open(fileobj=io.BytesIO(archive.read_bytes())) as tar:
            for member in tar:
                relative = PurePosixPath(*PurePosixPath(member.name).parts[1:])
                if not member.isfile() or relative.suffix != '.py':
                    continue
                total += member.size
                if member.size > 2_000_000 or total > 64_000_000:
                    raise ValueError('Snapshot excede limite.')
                dest = projection_path(root, relative)
                raw = tar.extractfile(member).read()
                if dest.exists() and dest.read_bytes() != raw:
                    partial = dest.read_bytes()
                    if len(partial) >= len(raw) or not raw.startswith(partial):
                        raise ValueError('Snapshot divergente.')
                    # Retomar somente prefixo da copia interrompida por disco cheio.
                    dest.write_bytes(raw)
                if not dest.exists():
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    dest.write_bytes(raw)
        tree = ast.parse((root / path).read_text(encoding='utf-8'))
        nodes = []
        def visit(body, prefix=''):
            for n in body:
                if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = prefix+n.name
                    if qualified == symbol:
                        nodes.append(n)
                    visit(n.body, qualified+'.')
        visit(tree.body)
        if len(nodes) != 1:
            raise ValueError(f'Entidade adjudicada nao univoca: {case_id}/{symbol}')
        original = row['problem_statement']
        additions = []
        # Somente arquivos de teste explicitamente citados no relato original.
        for test in sorted(set(re.findall(r'\btests/[\w/]+\.py', original))):
            source = projection_path(root, PurePosixPath(test))
            if source.is_file() and source.stat().st_size <= 8192:
                additions.append({'kind': 'PREEXISTING_TEST_SOURCE', 'path': test,
                    'text': source.read_text(encoding='utf-8'),
                    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'provenance': f"{row['base_commit']}:{test}", 'execution_verified': False})
        variants = input_variants(original, additions)
        cases.append({'id': case_id, 'split': split, 'category': 'BEHAVIORAL_MISMATCH',
            'repo': row['repo'], 'base_commit': row['base_commit'],
            'source_python_sha256': source_hash(root), 'source_archive_url': url,
            'source_archive_sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
            'problem_formulations': [original], 'inputs': variants,
            'pre_fix_evidence': additions,
            'original_reveals_location': bool(re.search(r'\.py(?:#L|:\d)|sessions\.py|models\.py', original)),
            'corrective_pr_url': f"https://github.com/{row['repo']}/pull/{case_id.rsplit('-',1)[1]}",
            'gold_patch_sha256': hashlib.sha256(row['patch'].encode()).hexdigest(),
            'oracle': {'root_path': path, 'root_symbol': symbol,
                'source_locations': [{'line': nodes[0].lineno, 'end_line': nodes[0].end_lineno}],
                'root_cause': cause, 'target_paths': [path],
                'repair_strategies': ['FIX_PRODUCER', 'CORRECT_RETURN_VALUE', 'CORRECT_CONTROL_FLOW', 'ERROR_HANDLING', 'VALIDATE_BOUNDARY', 'CORRECT_CONFIGURATION'],
                'abstention_expected': False, 'import_causal': False,
                'limitation': 'Location proxy; class attributes and multi-site changes may remain unrepresentable. Oracle never enters input modes.'}})
        print(f'Preparado: {case_id}', flush=True)
    raw = (json.dumps(cases, sort_keys=True, ensure_ascii=False, indent=2)+'\n').encode()
    target.mkdir(parents=True, exist_ok=True)
    (target/'cases.json').write_bytes(raw)
    (target/'manifest.json').write_text(json.dumps({'version': 16,
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'baseline': '4c259064f8761b0295ff846e45fa63ca21c61690',
        'cases_sha256': hashlib.sha256(raw).hexdigest(), 'independent_defects': 12,
        'development': 4, 'holdout': 8, 'budget_seconds': 240, 'memory_mb': 1024,
        'holdout_thresholds': {'completion_rate': 1.0, 'root_validity': .80,
            'hypothesis_location_precision': .75, 'unsupported_root_recommendations': 0},
        'policy': 'One-shot held-out matrix after final freeze; no holdout tuning.',
        'scope': '12 purposive Python projections; not production qualification.'}, indent=2)+'\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    target = Path('D:/JARVIS-evaluation-v16/external-v16/verified.json')
    if not target.exists():
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        rows = []
        for offset in range(0, 500, 100):
            url = ('https://datasets-server.huggingface.co/rows?dataset=princeton-nlp%2FSWE-bench_Verified'
                   f'&config=default&split=test&offset={offset}&length=100')
            with opener.open(url, timeout=30) as response:
                raw = response.read(20_000_001)
            if len(raw) > 20_000_000:
                raise ValueError('Dados excedem limite.')
            rows.extend(item['row'] for item in json.loads(raw)['rows'])
            print(f'Inventario: {len(rows)}', flush=True)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps({'source': 'princeton-nlp/SWE-bench_Verified', 'rows': rows}))
    rows = json.loads(target.read_text())['rows']
    historical = {case['id'] for version in (12, 13, 14, 15) for case in json.loads(Path(
        f'tests/data/predictive/external_v{version}/cases.json').read_text(encoding='utf-8'))}
    if args.seal:
        seal(rows, historical)
        return
    for row in rows:
        if row['instance_id'] not in historical and row['repo'] in {'psf/requests', 'pallets/flask', 'pytest-dev/pytest', 'sphinx-doc/sphinx'}:
            print(row['instance_id'], row['problem_statement'][:170].replace('\n', ' '))


if __name__ == '__main__':
    main()
