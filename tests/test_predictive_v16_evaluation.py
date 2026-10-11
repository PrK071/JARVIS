import hashlib
import json
from itertools import combinations
from pathlib import Path

from scripts.predictive_v16_evidence_study import evidence_information_gain, summarize
from scripts.predictive_v16_study import input_variants
from tern.orchestrator.predictive.external_evaluation import load_external_cases
from tern.orchestrator.predictive.semantic import pairwise_root_matrix, compare_root_candidates
from dataclasses import replace
from test_predictive_investigation import contract_case
from tern.orchestrator.predictive.causal import build_causal_slice


def test_presealed_population_and_legitimate_modes_only():
    corpus = Path('tests/data/predictive/external_v16')
    cases = load_external_cases(corpus, split='all')
    assert len(cases) == 12
    manifest = json.loads((corpus / 'manifest.json').read_text())
    assert manifest['cases_sha256'] == hashlib.sha256((corpus / 'cases.json').read_bytes()).hexdigest()
    assert sum(c['split'] == 'holdout' for c in cases) == 8
    for case in cases:
        assert case['inputs']['A']['text'] in case['inputs']['B']['text']
        assert case['inputs']['B']['new_information'] is False
        assert not case['inputs']['D']['available']
        assert case['inputs']['C']['available'] == bool(case['pre_fix_evidence'])


def test_value_of_information_is_state_delta_not_token_or_confidence_gain():
    before = {'api_ids': ['A1'], 'state': 'INSUFFICIENT', 'root_ids': ['R1']}
    same = evidence_information_gain(before, dict(before, confidence=1, extra_tokens=500))
    assert not same['changed']
    after = dict(before, api_ids=['A1', 'A2'], state='SUFFICIENT_FOR_INVESTIGATION')
    gain = evidence_information_gain(before, after)
    assert gain['changed']
    assert gain['added']['api_ids'] == ['A2']
    assert not gain['causal_correctness_improved']


def test_unavailable_modes_and_timeouts_preserve_denominators():
    cases = [{'id': 'one', 'inputs': {m: {'available': m in 'AB'} for m in 'ABCD'}}]
    result = summarize([{'case_id': 'one', 'input_mode': 'A', 'status': 'NOT_EVALUATED_TIMEOUT', 'wall_seconds': 240}], cases)
    assert result['A']['metrics']['completion_rate']['denominator'] == 1
    assert result['A']['metrics']['completion_rate']['numerator'] == 0
    assert result['A']['metrics']['root_correct']['rate'] is None
    assert result['D']['metrics']['completion_rate']['rate'] is None


def test_structuring_does_not_add_oracle_information():
    original = 'read() reports wrong result\nExpected behavior\ninteger\n```python\nread()\n```'
    variants = input_variants(original, [])
    assert original in variants['B']['text']
    assert not variants['C']['available'] and not variants['D']['available']


def test_cached_pairwise_matches_uncached_reference(tmp_path):
    context, _, snapshot, policy = contract_case(tmp_path, 'def read(flag):\n    if flag:\n        return None\n    return 1\n')
    slice_, roots = build_causal_slice(context, snapshot, policy)
    context = replace(context, causal_slice=slice_, root_cause_candidates=roots)
    roots = context.root_cause_candidates
    for ordering in (roots, tuple(reversed(roots))):
        reference = tuple(compare_root_candidates(a, b, context.causal_slice) for a, b in combinations(ordering, 2))
        assert pairwise_root_matrix(ordering, context.causal_slice) == reference
