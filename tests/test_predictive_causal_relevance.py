import ast

import pytest

from tern.orchestrator.predictive.import_graph import ImportCycleCause, ImportEdge, ImportSourceKind
from tern.orchestrator.predictive.relevance import (
    RelevanceStatus, SymptomKind, assess_import_cycle, describe_symptom,
)


def cycle():
    return ImportCycleCause(("pkg/a.py", "pkg/b.py"), (
        ImportEdge("pkg/a.py", "pkg/b.py", 1, "module", ImportSourceKind.PRODUCTION, symbol="b"),
        ImportEdge("pkg/b.py", "pkg/a.py", 1, "module", ImportSourceKind.PRODUCTION, symbol="a"),
    ), (ImportEdge("tests/test_a.py", "pkg/a.py", 1, "module", ImportSourceKind.TEST),))


def eager_trees():
    return {"pkg/a.py": ast.parse("from .b import b\na = 1\n"),
            "pkg/b.py": ast.parse("from .a import a\nb = 2\n")}


def test_behavioral_report_never_becomes_a_reported_exception():
    symptom = describe_symptom("api.decode returns bytes instead of text", None, None)
    assert symptom.kind is SymptomKind.BEHAVIORAL_MISMATCH
    assert symptom.reported_path is None
    assert symptom.expected_behavior is None  # Unknown contract is not invented.
    assert symptom.contract_evidence_ids == ()


def test_structural_cycle_without_symptom_bridge_is_not_causal():
    symptom = describe_symptom("api.decode returns wrong text", "pkg/a.py", 10)
    result = assess_import_cycle(cycle(), symptom, eager_trees())
    assert result.structural_relation_confirmed
    assert result.status is RelevanceStatus.INSUFFICIENT_CAUSAL_BRIDGE
    assert not result.relevance_to_symptom_confirmed
    assert not result.defect_origin_supported


@pytest.mark.parametrize("error", ("ImportError", "AttributeError"))
def test_observed_initialization_and_eager_lookup_support_import_origin(error):
    problem = f'Traceback\n  File "pkg/a.py", line 1, in <module>\n{error}: initialization failed'
    result = assess_import_cycle(cycle(), describe_symptom(problem, "pkg/a.py", 1), eager_trees())
    assert result.status is RelevanceStatus.DEFECT_ORIGIN_SUPPORTED
    assert result.bridge_refs


def test_test_observer_is_not_the_defect_entity():
    result = assess_import_cycle(cycle(), describe_symptom(
        'Traceback\n  File "tests/test_a.py", line 1, in <module>\nImportError: init',
        "tests/test_a.py", 1), eager_trees())
    assert result.entity_id == cycle().identity().id
    assert result.defect_origin_supported


def test_import_word_alone_does_not_prove_cycle_caused_symptom():
    result = assess_import_cycle(cycle(), describe_symptom(
        "ImportError is discussed but api.decode produces wrong text", None, None), eager_trees())
    assert not result.defect_origin_supported


def test_benign_module_cycle_without_eager_lookup_is_not_promoted():
    trees = {"pkg/a.py": ast.parse("from . import b\ndef run(): return b.run()\n"),
             "pkg/b.py": ast.parse("from . import a\ndef run(): return 2\n")}
    result = assess_import_cycle(cycle(), describe_symptom(
        'Traceback\n  File "pkg/a.py", line 1, in <module>\nImportError: missing',
        "pkg/a.py", 1), trees)
    assert not result.defect_origin_supported


def test_partially_initialized_attribute_report_can_anchor_without_traceback():
    symptom = describe_symptom("AttributeError: partially initialized module 'pkg.a' has no attribute 'a'", None, None)
    assert symptom.kind is SymptomKind.IMPORT_INITIALIZATION
    assert assess_import_cycle(cycle(), symptom, eager_trees()).defect_origin_supported
    other = describe_symptom("AttributeError: partially initialized module 'unrelated.z'", None, None)
    assert not assess_import_cycle(cycle(), other, eager_trees()).defect_origin_supported


def test_reported_location_alone_does_not_assert_exception():
    symptom = describe_symptom("pkg/a.py:10 produces an incorrect value", "pkg/a.py", 10)
    assert symptom.kind is SymptomKind.BEHAVIORAL_MISMATCH
    assert symptom.reported_path == "pkg/a.py"


@pytest.mark.parametrize("problem", (
    "ImportError: partially initialized module pkg.a while importing b",
    "tests/test_a.py imports pkg.a and observes the partially initialized production cycle.",
    "Import cycle while initializing pkg.a through pkg.b.",
))
def test_initialization_observations_do_not_require_traceback_or_quotes(problem):
    assert assess_import_cycle(cycle(), describe_symptom(problem, None, None), eager_trees()).defect_origin_supported


def test_merely_observing_scc_is_not_a_failure_report():
    problem = "Importing pkg.a from tests/test_a.py exposes an SCC; test is an observer."
    assert not assess_import_cycle(cycle(), describe_symptom(problem, None, None), eager_trees()).defect_origin_supported


def test_integrated_behavioral_anchor_does_not_connect_unrelated_import_cycle(tmp_path):
    from dataclasses import replace
    from tern.orchestrator.project_intelligence_v2 import ProjectIndexBuilderV2, ProjectCandidateGenerator
    from tern.orchestrator.predictive.service import build_problem_context
    from tern.orchestrator.predictive.grounding import build_evidence_ledger
    from tern.orchestrator.predictive.causal import build_causal_slice, CausalNodeKind
    from tern.orchestrator.security import PathPolicy

    (tmp_path/"pkg").mkdir()
    (tmp_path/"pkg"/"__init__.py").write_text("")
    (tmp_path/"pkg"/"a.py").write_text("from .b import b\na = 1\ndef decode(): return None\n")
    (tmp_path/"pkg"/"b.py").write_text("from .a import a\nb = 2\n")
    policy = PathPolicy((tmp_path,))
    snapshot = ProjectIndexBuilderV2(tmp_path, path_policy=policy).build()
    problem = "pkg/a.py::decode returns None instead of text; unrelated module pkg/b.py is present."
    context = build_problem_context(problem, snapshot, ProjectCandidateGenerator().generate(problem, snapshot), policy)
    context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
    result, roots = build_causal_slice(context, snapshot, policy)
    assert result.symptom.kind is SymptomKind.BEHAVIORAL_MISMATCH
    assert result.relevance_assessments
    assert all(not item.defect_origin_supported for item in result.relevance_assessments)
    assert all(root.import_scc is None for root in roots)
    if result.failure_site_id:
        site = next(node for node in result.nodes if node.id == result.failure_site_id)
        assert site.kind is CausalNodeKind.SYMPTOM_SITE
        assert "exception at" not in site.expression


def test_assessment_cannot_promote_origin_without_bridge():
    from tern.orchestrator.predictive.relevance import CausalRelevanceAssessment
    with pytest.raises(ValueError, match="ponte"):
        CausalRelevanceAssessment("SCC1", RelevanceStatus.DEFECT_ORIGIN_SUPPORTED, True, False, True)
