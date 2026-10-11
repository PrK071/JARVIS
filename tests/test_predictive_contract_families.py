import ast
from tern.orchestrator.predictive.testing.contract_families import contract_families
from tern.orchestrator.predictive.external_evaluation import load_external_cases
from pathlib import Path


def test_contract_oracle_families_are_static_and_reproducible():
    families = contract_families()
    assert families == contract_families()
    assert len(families) == 12
    assert len({f.name for f in families}) == 12
    for family in families:
        ast.parse(family.source)
    assert {f.expected for f in families} == {
        'EXPECTED_HYPOTHESIS', 'EXPECTED_CONFIRMED_ROOT', 'EXPECTED_ABSTENTION'}
    assert sum(f.violation for f in families) == 1


def test_external_v15_seal_and_independent_population():
    root = Path('tests/data/predictive')
    cases = load_external_cases(root / 'external_v15', split='all')
    historical = {c['id'] for version in (12, 13, 14)
                  for c in load_external_cases(root / f'external_v{version}', split='all')}
    assert len(cases) == 6
    assert not historical.intersection(c['id'] for c in cases)
    assert len({c['repo'] for c in cases}) == 3
    assert len(load_external_cases(root / 'external_v15', split='holdout')) == 3
    assert all(len(c['problem_formulations']) == 1 for c in cases)
