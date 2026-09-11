from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Iterable, Mapping

if TYPE_CHECKING:
    from .causal import CausalSlice, RepairStrategyKind, RootCauseCandidate


class RepairTargetKind(str, Enum):
    MODULE = "MODULE"
    FUNCTION = "FUNCTION"
    METHOD = "METHOD"
    PARAMETER = "PARAMETER"
    ATTRIBUTE = "ATTRIBUTE"
    RETURN_SITE = "RETURN_SITE"
    CALL_SITE = "CALL_SITE"
    CONFIG_VALUE = "CONFIG_VALUE"
    IMPORT_EDGE = "IMPORT_EDGE"
    TEST_EXPECTATION = "TEST_EXPECTATION"


def _target_id(*parts: object) -> str:
    raw = "\0".join(str(part or "") for part in parts).encode("utf-8")
    return "T" + hashlib.sha256(raw).hexdigest()[:12].upper()


@dataclass(frozen=True)
class RepairTarget:
    path: str
    scope_kind: RepairTargetKind
    symbol: str | None = None
    parameter: str | None = None
    attribute: str | None = None
    expression_id: str | None = None
    line: int | None = None

    def __post_init__(self) -> None:
        if not self.path.strip():
            raise ValueError("repair target path is required")
        if self.line is not None and self.line < 1:
            raise ValueError("repair target line must be positive")
        if self.scope_kind is RepairTargetKind.PARAMETER and not self.parameter:
            raise ValueError("parameter repair target requires parameter")
        if self.scope_kind is RepairTargetKind.ATTRIBUTE and not self.attribute:
            raise ValueError("attribute repair target requires attribute")

    @property
    def id(self) -> str:
        return _target_id(
            self.path, self.scope_kind.value, self.symbol, self.parameter,
            self.attribute, self.expression_id, self.line,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "id": self.id,
            "path": self.path,
            "scope_kind": self.scope_kind.value,
            "symbol": self.symbol,
            "parameter": self.parameter,
            "attribute": self.attribute,
            "expression_id": self.expression_id,
            "line": self.line,
        }


_CHILD_KINDS: Mapping[RepairTargetKind, frozenset[RepairTargetKind]] = {
    RepairTargetKind.MODULE: frozenset(RepairTargetKind),
    RepairTargetKind.FUNCTION: frozenset({
        RepairTargetKind.FUNCTION,
        RepairTargetKind.PARAMETER,
        RepairTargetKind.RETURN_SITE,
        RepairTargetKind.CALL_SITE,
    }),
    RepairTargetKind.METHOD: frozenset({
        RepairTargetKind.METHOD,
        RepairTargetKind.PARAMETER,
        RepairTargetKind.ATTRIBUTE,
        RepairTargetKind.RETURN_SITE,
        RepairTargetKind.CALL_SITE,
    }),
}


def target_refines(actual: RepairTarget, expected: RepairTarget) -> bool:
    """Whether actual names the expected scope or a deterministic child scope."""
    if actual.path != expected.path:
        return False
    if actual.scope_kind == expected.scope_kind:
        kind_ok = True
    else:
        kind_ok = actual.scope_kind in _CHILD_KINDS.get(expected.scope_kind, frozenset())
    if not kind_ok:
        return False
    expected_symbol = (expected.symbol or "").rsplit(".", 1)[-1]
    actual_symbol = (actual.symbol or "").rsplit(".", 1)[-1]
    if expected_symbol and actual_symbol and expected_symbol != actual_symbol:
        return False
    if expected.parameter and actual.parameter != expected.parameter:
        return False
    if expected.attribute and actual.attribute != expected.attribute:
        return False
    if expected.expression_id and actual.expression_id != expected.expression_id:
        return False
    if expected.line and actual.line and expected.line != actual.line:
        return False
    return True


def targets_compatible(actual: RepairTarget, expected: RepairTarget) -> bool:
    return target_refines(actual, expected) or target_refines(expected, actual)


def validate_repair_target(
    strategy: RepairStrategyKind,
    root: RootCauseCandidate,
    target: RepairTarget,
    causal_slice: CausalSlice,
) -> bool:
    from .causal import CausalNodeKind, RepairStrategyKind, RootCauseKind

    allowed_kinds = {
        RepairStrategyKind.CORRECT_ARGUMENT: {RepairTargetKind.CALL_SITE, RepairTargetKind.PARAMETER},
        RepairStrategyKind.CORRECT_RETURN_VALUE: {RepairTargetKind.RETURN_SITE, RepairTargetKind.FUNCTION, RepairTargetKind.METHOD},
        RepairStrategyKind.FIX_PRODUCER: {RepairTargetKind.FUNCTION, RepairTargetKind.METHOD, RepairTargetKind.PARAMETER, RepairTargetKind.ATTRIBUTE, RepairTargetKind.RETURN_SITE},
        RepairStrategyKind.FIX_CONSUMER_CONTRACT: {RepairTargetKind.FUNCTION, RepairTargetKind.METHOD, RepairTargetKind.PARAMETER},
        RepairStrategyKind.VALIDATE_BOUNDARY: {RepairTargetKind.FUNCTION, RepairTargetKind.METHOD, RepairTargetKind.PARAMETER, RepairTargetKind.CALL_SITE},
        RepairStrategyKind.CORRECT_CONTROL_FLOW: {RepairTargetKind.FUNCTION, RepairTargetKind.METHOD, RepairTargetKind.RETURN_SITE},
        RepairStrategyKind.CORRECT_CONFIGURATION: {RepairTargetKind.CONFIG_VALUE, RepairTargetKind.MODULE},
        RepairStrategyKind.CORRECT_IMPORT: {RepairTargetKind.IMPORT_EDGE},
        RepairStrategyKind.CORRECT_TEST_EXPECTATION: {RepairTargetKind.TEST_EXPECTATION},
        RepairStrategyKind.ERROR_HANDLING: {RepairTargetKind.FUNCTION, RepairTargetKind.METHOD},
        RepairStrategyKind.OTHER: set(RepairTargetKind),
    }
    if target.scope_kind not in allowed_kinds[strategy]:
        return False
    path_nodes = [node for node in causal_slice.nodes if node.id in root.causal_path]
    path_files = {node.path for node in path_nodes}
    if target.path not in path_files:
        return False
    if strategy is RepairStrategyKind.CORRECT_RETURN_VALUE:
        return any(
            node.path == target.path and node.kind is CausalNodeKind.RETURN_VALUE
            for node in path_nodes
        )
    if strategy is RepairStrategyKind.CORRECT_CONFIGURATION:
        return root.cause_kind is RootCauseKind.CONFIGURATION
    if strategy is RepairStrategyKind.CORRECT_IMPORT:
        return root.cause_kind is RootCauseKind.IMPORT_RESOLUTION
    return True


def best_target(targets: Iterable[RepairTarget]) -> RepairTarget | None:
    priority = {
        RepairTargetKind.RETURN_SITE: 0,
        RepairTargetKind.IMPORT_EDGE: 0,
        RepairTargetKind.CONFIG_VALUE: 0,
        RepairTargetKind.PARAMETER: 1,
        RepairTargetKind.ATTRIBUTE: 1,
        RepairTargetKind.CALL_SITE: 1,
        RepairTargetKind.FUNCTION: 2,
        RepairTargetKind.METHOD: 2,
        RepairTargetKind.TEST_EXPECTATION: 2,
        RepairTargetKind.MODULE: 3,
    }
    ordered = sorted(targets, key=lambda item: (priority[item.scope_kind], item.path, item.line or 0, item.id))
    return ordered[0] if ordered else None
