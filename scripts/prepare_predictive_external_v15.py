"""Corpus independente selado antes de alterar contratos/hipoteses."""
import ast
import io
import json
import tarfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from .prepare_predictive_external_v12 import digest, projection_path, source_hash, symbol_locations


ADJUDICATION = {
    'pallets__flask-4992': ('development', 'src/flask/config.py', 'Config', 'from_file',
        'RAISES_EXCEPTION', 'Config.from_file', None,
        'Text-only file opening cannot supply the binary reader required by the supplied TOML loader.'),
    'sphinx-doc__sphinx-7975': ('development', 'sphinx/environment/adapters/indexentries.py',
        'IndexEntries', 'create_index', 'RETURNS_VALUE', None, 2050,
        'The sort-key partition splits symbol entries into two groups and creates duplicate anchors.'),
    'sphinx-doc__sphinx-8506': ('development', 'sphinx/domains/std.py', 'Cmdoption',
        'handle_signature', 'RETURNS_VALUE', None, 8352,
        'Option parsing excludes bracket-leading syntax accepted by earlier versions; regex and normalization require correction.'),
    'sphinx-doc__sphinx-8627': ('holdout', 'sphinx/util/typing.py', None, 'restify',
        'RETURNS_VALUE', None, 8315,
        'Incorrect upstream Struct.__module__ metadata requires explicit qualification; stringify also changes.'),
    'sphinx-doc__sphinx-8801': ('holdout', 'sphinx/ext/autodoc/importer.py', None,
        'get_class_members', 'RETURNS_VALUE', None, 8800,
        'Annotation-only inherited attributes lose source documentation during ObjectMember construction.'),
    'pytest-dev__pytest-8906': ('holdout', 'src/_pytest/python.py', 'Module',
        '_importtestmodule', 'RAISES_EXCEPTION', None, None,
        'The collection error recommends a module marker that does not stop importing incompatible syntax.'),
}


def main():
    cache = Path('.orchestrator/external-v15')
    target = Path('tests/data/predictive/external_v15')
    if (target / 'manifest.json').exists():
        raise ValueError('Corpus ja selado.')
    dataset = json.loads(Path('.orchestrator/external-v12/dataset.json').read_text())
    rows = {row['instance_id']: row for row in dataset['rows']}
    old = {c['id'] for v in (12, 13, 14) for c in json.loads(Path(
        f'tests/data/predictive/external_v{v}/cases.json').read_text(encoding='utf-8'))}
    if old.intersection(ADJUDICATION):
        raise ValueError('Defeito historico reutilizado.')
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    cache.mkdir(parents=True, exist_ok=True)
    cases = []
    for case_id, (split, path, owner, name, effect, api, issue, cause) in ADJUDICATION.items():
        row = rows[case_id]
        url = f"https://codeload.github.com/{row['repo']}/tar.gz/{row['base_commit']}"
        archive = cache / f'{case_id}.tar.gz'
        if not archive.exists():
            with opener.open(url, timeout=120) as response:
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
                    raise ValueError('Snapshot divergente.')
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    dest.write_bytes(raw)
        sites = symbol_locations(ast.parse((root / path).read_text(encoding='utf-8')), name, owner)
        if len(sites) != 1:
            raise ValueError(f'Origem nao univoca: {case_id}')
        pr = f"https://github.com/{row['repo']}/pull/{case_id.rsplit('-', 1)[1]}"
        cases.append({'schema_version': 15, 'id': case_id, 'split': split,
            'category': 'BEHAVIORAL_MISMATCH', 'repo': row['repo'],
            'base_commit': row['base_commit'], 'dataset_revision': dataset['revision'],
            'source_archive_url': url, 'source_archive_sha256': digest(archive.read_bytes()),
            'source_python_sha256': source_hash(root), 'problem_formulations': [row['problem_statement']],
            'issue_url': f"https://github.com/{row['repo']}/issues/{issue}" if issue else pr,
            'corrective_pr_url': pr, 'gold_patch_sha256': digest(row['patch'].encode()),
            'gold_test_patch_sha256': digest(row['test_patch'].encode()),
            'oracle': {'root_path': path, 'root_symbol': f'{owner}.{name}' if owner else name,
                'source_locations': sites, 'root_cause': cause, 'affected_api': api,
                'behavioral_effect': effect, 'expected_relation': 'BEHAVIORAL_CONTRACT',
                'repair_strategies': ['FIX_PRODUCER', 'CORRECT_CONTROL_FLOW', 'CORRECT_RETURN_VALUE', 'ERROR_HANDLING'],
                'target_paths': [path], 'abstention_expected': False, 'import_causal': False,
                'adjudication_basis': 'Independent upstream production and test changes; original report. No corrective diff in prompts.',
                'limitation': 'Location proxy; 8627 also corrects stringify. Reports without a named production API must stay unanchored.'}})
    raw = (json.dumps(cases, sort_keys=True, ensure_ascii=False, indent=2) + '\n').encode()
    target.mkdir(parents=True, exist_ok=True)
    (target / 'cases.json').write_bytes(raw)
    manifest = {'version': 15, 'engine_commit': '76a15a4c8aeba3385a8d001903037defd0cbf99f',
        'frozen_at': datetime.now(timezone.utc).isoformat(), 'cases_sha256': digest(raw),
        'independent_defects': 6, 'development': 3, 'holdout': 3, 'formulations': 6,
        'repositories': ['pallets/flask', 'sphinx-doc/sphinx', 'pytest-dev/pytest'],
        'evaluator_version': 15, 'holdout_thresholds': {'completed_case_rate': 1.0,
            'hypothesis_location_precision': .75, 'hypothesis_evidence_validity': 1.0,
            'incorrect_hypothesis_promotion_rate': 0.0, 'contract_availability': .75,
            'correct_root_candidate_recall': .75, 'external_root_validity': .75},
        'policy': 'One-shot held-out evaluation after final runtime freeze. No tuning on historical defects.',
        'promotion_limit': 'Three held-out defects do not establish general external diagnostic readiness.'}
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest))


if __name__ == '__main__':
    main()
