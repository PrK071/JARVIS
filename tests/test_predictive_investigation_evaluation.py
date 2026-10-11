from tern.orchestrator.predictive.investigation_evaluation import investigation_metrics


def test_hypothesis_is_not_diagnosis_and_timeout_is_censored():
    row = {'mode': 'live', 'investigations': [{'oracle_location_match': True, 'evidence_valid': True}],
           'qwen': {'requests': 1}, 'contract_metrics_available': True,
           'oracle_contract_available': True, 'oracle_effect_bridge_demonstrated': False,
           'total_worker_seconds': 10}
    value = investigation_metrics([row], [{'status': 'NOT_EVALUATED_TIMEOUT'}])
    assert value['completed_case_rate']['rate'] == .5
    assert value['hypothesis_location_precision']['denominator'] == 1
    assert value['hypothesis_evidence_validity']['rate'] == 1
    assert value['effect_to_observation_recall']['rate'] == 0
    assert value['incorrect_hypothesis_promotion_rate']['rate'] is None
    assert 'external_root_validity' not in value
    assert value['p95_wall_time']['seconds'] == 10
    assert value['qwen_request_count'] == 1


def test_empty_metrics_are_explicitly_undefined():
    value = investigation_metrics([])
    for name in ('hypothesis_location_precision', 'contract_availability', 'completed_case_rate'):
        assert value[name]['rate'] is None
        assert value[name]['denominator'] == 0
        assert value[name]['population']
