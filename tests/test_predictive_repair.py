from __future__ import annotations

import pytest

from tern.orchestrator.predictive.repair import (
    RepairTarget,
    RepairTargetKind,
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
