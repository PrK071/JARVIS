import hashlib
from dataclasses import replace

import pytest

from tern.orchestrator.predictive.causal import build_causal_slice, CausalNodeKind
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.models import EvidenceExcerpt, EvidenceKind
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.predictive.symbols import SymbolRecoveryBudget, recover_symbol_context
from tern.orchestrator.project_intelligence_v2 import ProjectCandidateGenerator, ProjectIndexBuilderV2
from tern.orchestrator.security import PathPolicy


def prepare(tmp_path, files, problem, *, window=None):
    for name, source in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)
    policy = PathPolicy((tmp_path,))
    snapshot = ProjectIndexBuilderV2(tmp_path, path_policy=policy).build()
    context = build_problem_context(problem, snapshot, ProjectCandidateGenerator().generate(problem, snapshot), policy)
    if window:
        path, start, end = window
        lines = files[path].splitlines()
        context = replace(context, evidence=(EvidenceExcerpt(f"{path}:{start}-{end}", path, start, end,
                                                             "HARD", (), "\n".join(lines[start-1:end])),),
                          related_files=(path,))
    return context, snapshot, policy


def test_disjoint_methods_materialized_without_losing_initial_window(tmp_path):
    source = "class Api:\n    def unrelated(self):\n        return 1\n\n    def decode(self):\n        result = None\n        return result\n"
    context, snapshot, policy = prepare(tmp_path, {"api.py": source}, "api.py Api.decode() returns incorrect output", window=("api.py", 1, 3))
    context = recover_symbol_context(context, snapshot, policy)
    assert len(context.evidence) == 2
    assert context.relevant_symbol_contexts[0].qualified_name == "Api.decode"
    ledger = build_evidence_ledger(context, snapshot, policy)
    assert any(atom.kind is EvidenceKind.RETURN_STATEMENT and atom.symbol == "Api.decode" for atom in ledger.atoms)
    assert any(atom.kind is EvidenceKind.RETURN_STATEMENT and atom.symbol == "Api.unrelated" for atom in ledger.atoms)
    assert any(atom.kind is EvidenceKind.ASSIGNMENT and atom.start_line == 6 for atom in ledger.atoms)
    assert len({atom.id for atom in ledger.atoms}) == len(ledger.atoms)


def test_reported_api_is_anchor_not_proof_of_contract_or_defect(tmp_path):
    context, snapshot, policy = prepare(tmp_path, {"api.py": "def decode():\n    return None\n"}, "decode() gives unexpected output")
    recovered = recover_symbol_context(context, snapshot, policy)
    assert recovered.relevant_symbol_contexts[0].contract_evidence == ()
    assert recovered.relevant_symbol_contexts[0].as_dict()["status"] == "SYMBOL_RETRIEVED_NOT_PROVEN_DEFECTIVE"
    recovered = replace(recovered, evidence_ledger=build_evidence_ledger(recovered, snapshot, policy))
    causal, roots = build_causal_slice(recovered, snapshot, policy)
    assert not any(node.kind is CausalNodeKind.EXCEPTION_SITE for node in causal.nodes)
    assert roots


def test_ambiguous_receiver_does_not_invent_unique_target(tmp_path):
    source = "class One:\n    def decode(self):\n        return 1\nclass Two:\n    def decode(self):\n        return 2\n"
    context, snapshot, policy = prepare(tmp_path, {"api.py": source}, "api.py receiver.decode() has incorrect output")
    recovered = recover_symbol_context(context, snapshot, policy)
    assert {item.qualified_name for item in recovered.relevant_symbol_contexts} == {"One.decode", "Two.decode"}
    assert all(item.selection_reason == "REPORTED_API_AMBIGUOUS" for item in recovered.relevant_symbol_contexts)


def test_comments_and_docstrings_do_not_form_structural_neighbors(tmp_path):
    source = 'def decode():\n    """Call private_producer and return validation."""\n    # private_producer() is not executed\n    return 1\ndef private_producer():\n    return None\n'
    context, snapshot, policy = prepare(tmp_path, {"api.py": source}, "decode() returns incorrectly")
    recovered = recover_symbol_context(context, snapshot, policy)
    assert [item.qualified_name for item in recovered.relevant_symbol_contexts] == ["decode"]


def test_resolved_return_producer_depth_and_files_are_bounded(tmp_path):
    files = {"api.py": "from backend import produce\ndef decode():\n    return produce()\n",
             "backend.py": "from origin import fetch\ndef produce():\n    return fetch()\n",
             "origin.py": "def fetch():\n    return None\n"}
    context, snapshot, policy = prepare(tmp_path, files, "api.py decode() returns incorrectly", window=("api.py", 1, 3))
    one = recover_symbol_context(context, snapshot, policy, budget=SymbolRecoveryBudget(max_extra_files=1))
    assert set(one.related_files) == {"api.py", "backend.py"}
    two = recover_symbol_context(context, snapshot, policy)
    assert len(set(two.related_files) - set(context.related_files)) <= 2
    assert {item.qualified_name for item in two.relevant_symbol_contexts} == {"decode", "produce", "fetch"}
    assert max(item.depth for item in two.relevant_symbol_contexts) == 2


def test_no_anchor_no_expansion_true_insufficiency(tmp_path):
    context, snapshot, policy = prepare(tmp_path, {"api.py": "def decode():\n    return None\n"}, "api.py mysterious behavior")
    recovered = recover_symbol_context(context, snapshot, policy)
    assert not recovered.relevant_symbol_contexts
    assert recovered.evidence == context.evidence
    disabled = recover_symbol_context(context, snapshot, policy, budget=SymbolRecoveryBudget(max_attempts=0))
    assert disabled is context


def test_budget_validation_determinism_and_read_only(tmp_path):
    context, snapshot, policy = prepare(tmp_path, {"api.py": "def decode():\n    return None\n"}, "decode() wrong output")
    before = hashlib.sha256((tmp_path / "api.py").read_bytes()).hexdigest()
    one = recover_symbol_context(context, snapshot, policy)
    two = recover_symbol_context(context, snapshot, policy)
    assert one == two
    assert before == hashlib.sha256((tmp_path / "api.py").read_bytes()).hexdigest()
    limited = recover_symbol_context(context, snapshot, policy, budget=SymbolRecoveryBudget(max_context_bytes=1))
    assert not limited.relevant_symbol_contexts
    with pytest.raises(ValueError):
        SymbolRecoveryBudget(max_attempts=2)
    with pytest.raises(ValueError):
        SymbolRecoveryBudget(max_extra_files=-1)


def test_dynamic_receiver_not_resolved_by_final_name(tmp_path):
    files = {"api.py": "from backend import decode\ndef process(receiver):\n    return receiver.decode()\n",
             "backend.py": "def decode():\n    return None\n"}
    context, snapshot, policy = prepare(tmp_path, files, "api.py process() returns incorrectly", window=("api.py", 1, 3))
    recovered = recover_symbol_context(context, snapshot, policy)
    assert [item.qualified_name for item in recovered.relevant_symbol_contexts] == ["process"]


def test_import_alias_is_structural_but_shadowed_name_is_not(tmp_path):
    files = {"api.py": "from backend import produce as fetch\ndef decode():\n    return fetch()\ndef process(fetch):\n    return fetch()\n",
             "backend.py": "def produce():\n    return None\n"}
    context, snapshot, policy = prepare(tmp_path, files, "api.py decode() wrong output", window=("api.py", 1, 3))
    recovered = recover_symbol_context(context, snapshot, policy)
    assert {item.qualified_name for item in recovered.relevant_symbol_contexts} == {"decode", "produce"}
    shadowed = recover_symbol_context(replace(context, problem="api.py process() wrong output"), snapshot, policy)
    assert [item.qualified_name for item in shadowed.relevant_symbol_contexts] == ["process"]


def test_duplicate_definition_identity_and_symbol_budget(tmp_path):
    source = "def decode():\n    return 1\ndef decode():\n    return 2\n"
    context, snapshot, policy = prepare(tmp_path, {"api.py": source}, "decode() wrong output")
    recovered = recover_symbol_context(context, snapshot, policy)
    assert len(recovered.relevant_symbol_contexts) == 2
    assert len({item.symbol_id for item in recovered.relevant_symbol_contexts}) == 2
    limited = recover_symbol_context(context, snapshot, policy, budget=SymbolRecoveryBudget(max_extra_symbols=1))
    assert len(limited.relevant_symbol_contexts) == 1


def test_value_read_not_comment_or_constant_text(tmp_path):
    source = 'def actual():\n    return settings.TIMEOUT\ndef distractor():\n    """settings.TIMEOUT"""\n    text = "settings.TIMEOUT"\n    return text\n'
    context, snapshot, policy = prepare(tmp_path, {"api.py": source}, "api.py settings.TIMEOUT has an incorrect value")
    recovered = recover_symbol_context(context, snapshot, policy)
    assert [item.qualified_name for item in recovered.relevant_symbol_contexts] == ["actual"]
    assert recovered.relevant_symbol_contexts[0].selection_reason == "REPORTED_VALUE_READ"


def test_return_annotation_is_fact_not_verified_violation(tmp_path):
    context, snapshot, policy = prepare(tmp_path, {"api.py": "def decode() -> str:\n    return 'ok'\n"}, "decode() output")
    recovered = recover_symbol_context(context, snapshot, policy)
    assert recovered.relevant_symbol_contexts[0].contract_evidence == ("AST_RETURN_ANNOTATION",)
    assert "defect" not in " ".join(recovered.relevant_symbol_contexts[0].structural_anchors)


def test_ledger_lookup_index_preserves_original_order_and_fallback():
    from tern.orchestrator.predictive.causal import _evidence_at
    from tern.orchestrator.predictive.models import EvidenceAtom, EvidenceLedger
    atoms = tuple(EvidenceAtom(f"E{index}", EvidenceKind.ASSIGNMENT, path, line, line,
                               "fato AST", .8)
                  for index, (path, line) in enumerate((('api.py', 3), ('other.py', 3),
                                                      ('api.py', 3), ('api.py', 8))))
    ledger = EvidenceLedger(atoms)
    for path, line in (('api.py', 3), ('api.py', 99), ('missing.py', 1)):
        expected = tuple(atom.id for atom in atoms if atom.path == path
                         and atom.start_line <= line <= atom.end_line)
        expected = expected or tuple(atom.id for atom in atoms if atom.path == path)[:2]
        assert _evidence_at(ledger, path, line) == expected
    assert ledger.for_path('api.py') == (atoms[0], atoms[2], atoms[3])
    assert ledger.get('E2') is atoms[2]
    assert ledger.get('missing') is None
    assert ledger.as_dict()['atoms'] == [atom.as_dict() for atom in atoms]


def test_symbol_context_serialization_is_additive_and_read_only(tmp_path):
    from tern.orchestrator.predictive.service import PredictiveDecisionService
    prepare(tmp_path, {'api.py': 'def decode():\n    return None\n'}, 'decode() incorrect output')
    report = PredictiveDecisionService(None, path_policy=PathPolicy((tmp_path,))).predict(
        'decode() incorrect output', tmp_path)
    raw = report.as_dict()
    assert raw['schema_version'] == 6
    assert raw['symbol_context_schema_version'] == 1
    assert raw['relevant_symbol_contexts'][0]['qualified_name'] == 'decode'
    assert raw['execution_authorized'] is False
    assert raw['requires_approval'] is True
