"""Presenca estrutural, suporte causal e evidencia faltante nao sao sinonimos."""
from dataclasses import dataclass
from enum import Enum

from .causal import CausalNodeKind
from .investigation import build_investigable_hypotheses, validate_hypothesis_promotion
from .relevance import SymptomKind, describe_symptom


class EvidenceSufficiency(str, Enum):
    SUFFICIENT_FOR_DIAGNOSIS = 'SUFFICIENT_FOR_DIAGNOSIS'
    SUFFICIENT_FOR_INVESTIGATION = 'SUFFICIENT_FOR_INVESTIGATION'
    REQUIRES_RUNTIME_OBSERVATION = 'REQUIRES_RUNTIME_OBSERVATION'
    REQUIRES_CONTRACT_EVIDENCE = 'REQUIRES_CONTRACT_EVIDENCE'
    REQUIRES_API_RESOLUTION = 'REQUIRES_API_RESOLUTION'
    AMBIGUOUS = 'AMBIGUOUS'
    INSUFFICIENT = 'INSUFFICIENT'


@dataclass(frozen=True)
class DiagnosticEvidenceRequest:
    missing_fact: str
    why_needed: str
    related_symbols: tuple[str, ...]
    acceptable_evidence_types: tuple[str, ...]
    disambiguates: tuple[str, ...]

    def as_dict(self):
        return dict(self.__dict__, related_symbols=list(self.related_symbols),
            acceptable_evidence_types=list(self.acceptable_evidence_types),
            disambiguates=list(self.disambiguates), execution_requested=False)


@dataclass(frozen=True)
class RootSupportAssessment:
    root_id: str
    state: str
    reasons: tuple[str, ...]
    evidence_ids: tuple[str, ...]

    def as_dict(self):
        return dict(self.__dict__, reasons=list(self.reasons), evidence_ids=list(self.evidence_ids))


@dataclass(frozen=True)
class EvidenceSufficiencyAssessment:
    classification: EvidenceSufficiency
    roots: tuple[RootSupportAssessment, ...]
    requests: tuple[DiagnosticEvidenceRequest, ...]
    funnel: tuple[dict, ...]
    investigation_eligible: bool
    observation_identified: bool
    api_resolved: bool
    contract_available: bool

    @property
    def supported_root_ids(self):
        return tuple(root.root_id for root in self.roots if root.state == 'ROOT_SUPPORTED')

    def as_dict(self):
        return dict(classification=self.classification.value,
            roots=[r.as_dict() for r in self.roots], requests=[r.as_dict() for r in self.requests],
            funnel=list(self.funnel), investigation_eligible=self.investigation_eligible,
            observation_identified=self.observation_identified, api_resolved=self.api_resolved,
            contract_available=self.contract_available, supported_root_ids=list(self.supported_root_ids),
            limitation='Suporte estatico limitado nao comprova execucao nem correcao universal.')


def assess_evidence_sufficiency(context, contracts, costs=None):
    """Somente fatos do runtime; oracle, score e confianca nao participam."""
    slice_ = context.causal_slice
    roots = context.root_cause_candidates
    graph = context.behavioral_graph
    ledger = context.evidence_ledger
    symptom = slice_.symptom if slice_ and slice_.symptom else describe_symptom(context.problem, None, None)
    nodes = {n.id: n for n in slice_.nodes} if slice_ else {}
    pairs = {(e.source_id, e.target_id) for e in slice_.edges if e.deterministic} if slice_ else set()
    locations = build_investigable_hypotheses(context, contracts)
    contracts_by_id = {c.id: c for c in contracts}
    support = []
    for root in roots:
        ids_valid = bool(root.evidence_ids) and all(ledger.get(i) for i in root.evidence_ids)
        path_valid = bool(root.causal_path) and all(i in nodes for i in root.causal_path) and all(
            (a, b) in pairs for a, b in zip(root.causal_path, root.causal_path[1:]))
        supported = False
        reasons = []
        if not ids_valid or not path_valid:
            reasons.append('CAUSAL_PATH_OR_EVIDENCE_MISSING')
        elif root.import_scc:
            supported = any(a.entity_id == root.import_scc.id and a.defect_origin_supported
                for a in slice_.relevance_assessments)
            reasons.append('IMPORT_RELEVANCE_VALIDATION' if supported else 'IMPORT_BRIDGE_MISSING')
        elif symptom.kind is SymptomKind.EXCEPTION_FAILURE:
            failure = nodes.get(slice_.failure_site_id)
            supported = bool(failure and failure.kind is CausalNodeKind.EXCEPTION_SITE
                and root.causal_path[-1] == failure.id and symptom.reported_line == failure.line
                and symptom.reported_path == failure.path)
            reasons.append('OBSERVED_FRAME_AND_STRUCTURAL_LINEAGE' if supported else 'EXCEPTION_BRIDGE_MISSING')
        else:
            for h in locations:
                contract = contracts_by_id[h.contract_id]
                if validate_hypothesis_promotion(h, contract, root, slice_, ledger).approved:
                    supported = True
                    break
            reasons.append('STATIC_CONTRACT_AND_CAUSAL_RELATION' if supported else 'BEHAVIORAL_VIOLATION_NOT_DEMONSTRATED')
        has_contract = any(c.path == root.origin_path and c.symbol == root.origin_symbol
            and c.contract_exists for c in contracts)
        state = ('ROOT_SUPPORTED' if supported else 'ROOT_EXISTS_BUT_CONTRACT_UNVERIFIED'
            if has_contract else 'ROOT_EXISTS_BUT_CAUSAL_BRIDGE_MISSING')
        support.append(RootSupportAssessment(root.id, state, tuple(reasons), root.evidence_ids))
    supported = any(r.state == 'ROOT_SUPPORTED' for r in support)
    observed = symptom.kind is not SymptomKind.UNRESOLVED
    api = bool(graph and graph.entry_ids)
    requests = []
    related = tuple(dict.fromkeys(h.symbol for h in locations))
    alternatives = tuple(r.root_id for r in support if r.state != 'ROOT_SUPPORTED')
    if not supported and observed:
        if not api and symptom.kind is not SymptomKind.EXCEPTION_FAILURE:
            requests.append(DiagnosticEvidenceRequest('AFFECTED_API_INVOCATION',
                'Resolver a entrada afetada antes de atribuir responsabilidade aos roots existentes.', (),
                ('REPRODUCIBLE_API_CALL', 'REAL_TRACEBACK'), alternatives or ('ROOT_MISSING',)))
        if locations:
            requests.append(DiagnosticEvidenceRequest('EFFECT_TO_OBSERVATION',
                'Distinguir operacao alcancavel de operacao que produziu o comportamento relatado.', related,
                ('EXISTING_EXECUTION_TRACE', 'INITIAL_AND_FINAL_STATE', 'INPUT_OUTPUT_PAIR'), alternatives or ('INVESTIGATION_ONLY',)))
            if any(not c.violation_supported for c in contracts):
                requests.append(DiagnosticEvidenceRequest('APPLICABLE_BEHAVIORAL_CONTRACT',
                    'Distinguir comportamento permitido de violacao no efeito identificado.', related,
                    ('PREEXISTING_REGRESSION_ASSERTION', 'EXPECTED_AND_OBSERVED_OUTPUT'), alternatives or ('INVESTIGATION_ONLY',)))
    classification = (EvidenceSufficiency.SUFFICIENT_FOR_DIAGNOSIS if supported else
        EvidenceSufficiency.INSUFFICIENT if not observed else
        EvidenceSufficiency.SUFFICIENT_FOR_INVESTIGATION if locations else
        EvidenceSufficiency.REQUIRES_API_RESOLUTION if not api else
        EvidenceSufficiency.REQUIRES_CONTRACT_EVIDENCE if not any(c.violation_supported for c in contracts) else
        EvidenceSufficiency.REQUIRES_RUNTIME_OBSERVATION)
    costs = costs or {}
    stages = (
        ('Problem', ('USER_REPORT',), (), (), 'initial_retrieval'),
        ('Observed Symptom', (symptom.kind.value,), (), (), 'initial_retrieval'),
        ('Relevant Files', context.related_files, (), (), 'initial_retrieval'),
        ('Relevant APIs', tuple(graph.entry_ids) if graph else (), (),
            tuple(u.expression for u in graph.unresolved) if graph else ('API_UNRESOLVED',), 'symbol_resolution'),
        ('Implementation Graph', tuple(c.id for c in graph.callables) if graph else (),
            tuple(r.id for r in graph.relations) if graph else (), (), 'implementation_graph'),
        ('Evidence Ledger', tuple(a.id for a in ledger.atoms), (), (), 'ledger'),
        ('Contract Evidence', tuple(c.id for c in contracts),
            tuple(i for c in contracts for i in c.evidence_ids),
            tuple(c.id for c in contracts if not c.violation_supported), 'behavioral_contracts'),
        ('Root Candidates', tuple(r.id for r in roots), tuple(i for r in roots for i in r.evidence_ids), (), 'causal_slice'),
        ('Candidate Support', tuple(r.root_id for r in support if r.state == 'ROOT_SUPPORTED'), (),
            tuple(r.root_id for r in support if r.state != 'ROOT_SUPPORTED'), 'evidence_sufficiency'))
    funnel = tuple({'stage': name, 'entities': list(entities), 'evidence_or_relations': list(relations),
        'unresolved': list(unknown), 'loss_reason': ('ROOT_MISSING' if name == 'Root Candidates' and not entities
            else 'ROOT_PRESENT_BUT_UNSUPPORTED' if name == 'Candidate Support' and roots and not entities
            else None), 'seconds': costs.get(phase),
        'next_evidence': [r.missing_fact for r in requests] if unknown or not entities else []}
        for name, entities, relations, unknown, phase in stages)
    return EvidenceSufficiencyAssessment(classification, tuple(support), tuple(requests), funnel,
        bool(locations) and not supported, observed, api, any(c.contract_exists for c in contracts))
