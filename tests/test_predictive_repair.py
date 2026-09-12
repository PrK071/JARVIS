from __future__ import annotations

import pytest

from tern.orchestrator.predictive.causal import (
    RepairStrategyKind,
    RootCauseCandidate,
    RootCauseKind,
)
from tern.orchestrator.predictive.repair import (
    RepairTarget,
    RepairTargetKind,
    best_target,
    target_refines,
    targets_compatible,
)


def test_repair_target_ids_are_stable_and_scope_sensitive():
    first = RepairTarget("pkg/math.py", RepairTargetKind.PARAMETER, "mean", "values", line=4)
    same = RepairTarget("pkg/math.py", RepairTargetKind.PARAMETER, "mean", "values", line=4)
    wider = RepairTarget("pkg/math.py", RepairTargetKind.FUNCTION, "mean", line=4)

    assert first.id == same.id
    assert first.id != wider.id


def test_parameter_refines_its_containing_function():
    parameter = RepairTarget("pkg/math.py", RepairTargetKind.PARAMETER, "mean", "values", line=4)
    function = RepairTarget("pkg/math.py", RepairTargetKind.FUNCTION, "mean", line=4)

    assert target_refines(parameter, function)
    assert targets_compatible(function, parameter)


def test_unrelated_parameter_is_not_compatible():
    actual = RepairTarget("pkg/math.py", RepairTargetKind.PARAMETER, "mean", "count", line=4)
    expected = RepairTarget("pkg/math.py", RepairTargetKind.PARAMETER, "mean", "values", line=4)

    assert not targets_compatible(actual, expected)


def test_parameter_and_attribute_require_names():
    with pytest.raises(ValueError, match="requires parameter"):
        RepairTarget("pkg/math.py", RepairTargetKind.PARAMETER)
    with pytest.raises(ValueError, match="requires attribute"):
        RepairTarget("pkg/math.py", RepairTargetKind.ATTRIBUTE)


def test_best_target_uses_strategy_and_causal_origin():
    root = RootCauseCandidate(
        "R1", RootCauseKind.ARGUMENT_BINDING,
        "pkg/math.py", "values", 5, "pkg/math.py", 2,
        ("N1", "N2"), ("E1",), 0.8, 1.0, 1, 1.0, 0.8, "flow",
    )
    parameter = RepairTarget(
        "pkg/math.py", RepairTargetKind.PARAMETER, "mean", "values", line=5
    )
    call_site = RepairTarget(
        "pkg/math.py", RepairTargetKind.CALL_SITE, "mean", line=6
    )

    assert best_target(
        (call_site, parameter),
        strategy=RepairStrategyKind.VALIDATE_BOUNDARY,
        root=root,
    ) == parameter
    assert best_target(
        (parameter, call_site),
        strategy=RepairStrategyKind.CORRECT_ARGUMENT,
        root=root,
    ) == call_site
