"""Populacoes de investigacao sao separadas de diagnosticos confirmados."""
from .external_evaluation import rate


def investigation_metrics(rows, incomplete=()):
    live = [r for r in rows if r['mode'] == 'live']
    hypotheses = [h for r in live for h in r.get('investigations', ())]
    contracts = [r for r in live if r.get('contract_metrics_available')]
    validations = [v for r in live for v in r.get('promotion_validations', ())]
    walls = sorted(r['total_worker_seconds'] for r in live if r.get('total_worker_seconds') is not None)
    def percentile(values, p):
        if not values:
            return None
        position = (len(values)-1)*p
        low = int(position)
        high = min(low+1, len(values)-1)
        return values[low] + (values[high]-values[low])*(position-low)
    return {
        'investigable_hypothesis_coverage': rate(sum(bool(r.get('investigations')) for r in live),
            len(live), 'completed live formulations; independent from recommendation coverage'),
        'hypothesis_location_precision': rate(sum(h['oracle_location_match'] for h in hypotheses),
            len(hypotheses), 'selected investigable hypotheses; source-location proxy, not confirmed causes'),
        'hypothesis_evidence_validity': rate(sum(h['evidence_valid'] for h in hypotheses),
            len(hypotheses), 'selected hypotheses with exact ledger identities'),
        'contract_availability': rate(sum(r['oracle_contract_available'] for r in contracts),
            len(contracts), 'instrumented completed formulations; identified source or reported expectation'),
        'effect_to_observation_recall': rate(sum(r['oracle_effect_bridge_demonstrated'] for r in contracts),
            len(contracts), 'instrumented formulations; inferred reachability excluded'),
        'incorrect_hypothesis_promotion_rate': rate(sum(v['approved'] and not v['oracle_match'] for v in validations),
            sum(v['approved'] for v in validations), 'approved deterministic transitions only; empty is undefined'),
        'completed_case_rate': rate(len(rows), len(rows)+len(incomplete), 'attempted formulations; censored workers excluded from diagnostic metrics'),
        'timeout_rate': rate(sum(r['status'] == 'NOT_EVALUATED_TIMEOUT' for r in incomplete),
            len(rows)+len(incomplete), 'attempted formulations'),
        'median_wall_time': {'seconds': percentile(walls, .5), 'denominator': len(walls),
            'population': 'completed worker wall time including preflight; censored durations excluded'},
        'p95_wall_time': {'seconds': percentile(walls, .95), 'denominator': len(walls),
            'population': 'completed worker wall time including preflight; small samples descriptive only'},
        'qwen_request_count': sum(r['qwen']['requests'] for r in live),
    }
