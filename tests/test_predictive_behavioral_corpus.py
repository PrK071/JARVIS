import ast
from pathlib import Path

from tern.orchestrator.predictive.external_evaluation import load_external_cases
from tern.orchestrator.predictive.testing.behavioral_families import behavioral_families


def test_reproducible_positive_and_negative_development_controls():
    first, second = behavioral_families(), behavioral_families()
    assert first == second
    assert len(first) == 12 and len({f.name for f in first}) == 12
    assert any(f.implementation is None for f in first)
    assert any(f.effect is None for f in first)
    for family in first:
        for source in family.files.values():
            ast.parse(source)


def test_new_sealed_defects_do_not_reuse_observed_holdouts():
    current = load_external_cases(Path('tests/data/predictive/external_v14'), split='all')
    old = {case['id'] for version in (12, 13) for case in load_external_cases(
        Path(f'tests/data/predictive/external_v{version}'), split='all')}
    assert not old.intersection(c['id'] for c in current)
    assert len(current) == 4
