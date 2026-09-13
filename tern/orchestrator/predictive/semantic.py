from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .causal import (
    CausalEdgeKind,
    CausalNode,
    CausalNodeKind,
    CausalSlice,
    RepairStrategyKind,
    RootCauseCandidate,
    RootCauseKind,
)
from .repair import RepairTarget, RepairTargetKind


class CausalRole(str, Enum):
    PRODUCER = "PRODUCER"
    CONSUMER = "CONSUMER"
    CALLER = "CALLER"
    CALLEE = "CALLEE"
    RETURN_SOURCE = "RETURN_SOURCE"
    ARGUMENT_SOURCE = "ARGUMENT_SOURCE"
    CONFIG_SOURCE = "CONFIG_SOURCE"
    IMPORT_SOURCE = "IMPORT_SOURCE"
    FAILURE_SITE = "FAILURE_SITE"
    INTERMEDIATE_WRAPPER = "INTERMEDIATE_WRAPPER"


class OriginRole(str, Enum):
    SOURCE = "SOURCE"
    BOUNDARY = "BOUNDARY"
    INTERMEDIARY = "INTERMEDIARY"
    MANIFESTATION = "MANIFESTATION"
    UNKNOWN = "UNKNOWN"


class RelationToFailure(str, Enum):
    DIRECT = "DIRECT"
    UPSTREAM = "UPSTREAM"
    INTERMEDIATE = "INTERMEDIATE"
    MANIFESTATION = "MANIFESTATION"
    UNKNOWN = "UNKNOWN"


class ContractRole(str, Enum):
    RETURN = "RETURN"
    INPUT = "INPUT"
    CONFIGURATION = "CONFIGURATION"
    IMPORT = "IMPORT"
    STATE = "STATE"
    NONE = "NONE"


class PairwiseRootReason(str, Enum):
    DIRECT_CAUSAL_ORIGIN = "DIRECT_CAUSAL_ORIGIN"
    UPSTREAM_CONTRACT_ORIGIN = "UPSTREAM_CONTRACT_ORIGIN"
    MANIFESTATION_ONLY = "MANIFESTATION_ONLY"
    INTERMEDIATE_WRAPPER = "INTERMEDIATE_WRAPPER"
    STRONGER_STRUCTURAL_PATH = "STRONGER_STRUCTURAL_PATH"
    STRONGER_TEST_SUPPORT = "STRONGER_TEST_SUPPORT"
    INSUFFICIENT_TO_DISTINGUISH = "INSUFFICIENT_TO_DISTINGUISH"


class TargetRelation(str, Enum):
    ROOT = "ROOT"
    ROOT_SCOPE = "ROOT_SCOPE"
    CAUSAL_PATH = "CAUSAL_PATH"
    FAILURE = "FAILURE"
    UNRELATED = "UNRELATED"


class TargetPreference(str, Enum):
    PREFERRED = "PREFERRED"
    COMPATIBLE = "COMPATIBLE"
    INVALID = "INVALID"


@dataclass(frozen=True)
class RootCauseSignature:
    cause_kind: RootCauseKind
    causal_roles: tuple[CausalRole, ...]
    origin_role: OriginRole
    relation_to_failure: RelationToFailure
    contract_role: ContractRole

    def as_dict(self) -> dict[str, object]:
        return {
            "cause_kind": self.cause_kind.value,
            "causal_roles": [item.value for item in self.causal_roles],
            "origin_role": self.origin_role.value,
            "relation_to_failure": self.relation_to_failure.value,
            "contract_role": self.contract_role.value,
        }


@dataclass(frozen=True)
class RootPairwiseComparison:
    preferred_id: str | None
    rejected_id: str | None
    reason: PairwiseRootReason
    valid: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "preferred_id": self.preferred_id,
            "rejected_id": self.rejected_id,
            "reason": self.reason.value,
            "valid": self.valid,
        }


@dataclass(frozen=True)
class RepairTargetSignature:
    target_kind: RepairTargetKind
    causal_roles: tuple[CausalRole, ...]
    owning_symbol_role: OriginRole
    relation_to_root: TargetRelation
    relation_to_failure: TargetRelation
    causal_distance: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "target_kind": self.target_kind.value,
            "causal_roles": [item.value for item in self.causal_roles],
            "owning_symbol_role": self.owning_symbol_role.value,
            "relation_to_root": self.relation_to_root.value,
            "relation_to_failure": self.relation_to_failure.value,
            "causal_distance": self.causal_distance,
        }


def _origin_node(root: RootCauseCandidate, causal_slice: CausalSlice) -> CausalNode | None:
    if not root.causal_path:
        return None
    return next((item for item in causal_slice.nodes if item.id == root.causal_path[0]), None)


def _incoming(causal_slice: CausalSlice) -> Mapping[str, tuple[object, ...]]:
    values: dict[str, list[object]] = {}
    for edge in causal_slice.edges:
        values.setdefault(edge.target_id, []).append(edge)
    return {key: tuple(sorted(items, key=lambda item: item.id)) for key, items in values.items()}


def _transparent_expression(expression: str) -> bool:
    value = expression.strip()
    if not value.startswith("return "):
        return False
    returned = value[7:].strip()
    if not returned:
        return False
    if returned.replace(".", "").replace("_", "").isalnum():
        return True
    if returned.endswith(")") and "(" in returned:
        prefix = returned.split("(", 1)[0]
        return prefix.replace(".", "").replace("_", "").isalnum()
    return False


def is_transparent_wrapper(root: RootCauseCandidate, causal_slice: CausalSlice) -> bool:
    node = _origin_node(root, causal_slice)
    if node is None or node.kind is not CausalNodeKind.RETURN_VALUE:
        return False
    if not _transparent_expression(node.expression):
        return False
    nodes = {item.id: item for item in causal_slice.nodes}
    incoming = _incoming(causal_slice)
    queue: deque[tuple[str, int]] = deque([(node.id, 0)])
    visited = {node.id}
    allowed = {
        CausalEdgeKind.ASSIGNED_FROM,
        CausalEdgeKind.RETURNED_FROM,
        CausalEdgeKind.CALLS,
    }
    while queue:
        current, depth = queue.popleft()
        if depth >= 5:
            continue
        for edge in incoming.get(current, ()):
            if not edge.deterministic or edge.kind not in allowed or edge.source_id in visited:
                continue
            visited.add(edge.source_id)
            source = nodes[edge.source_id]
            if (
                source.kind in {CausalNodeKind.RETURN_VALUE, CausalNodeKind.LITERAL}
                and (source.path, source.scope) != (node.path, node.scope)
            ):
                return True
            queue.append((source.id, depth + 1))
    return False


def root_cause_signature(
    root: RootCauseCandidate, causal_slice: CausalSlice
) -> RootCauseSignature:
    node = _origin_node(root, causal_slice)
    roles: set[CausalRole] = set()
    at_failure = (
        root.origin_path == root.failure_path and root.origin_line == root.failure_line
    )
    transparent = is_transparent_wrapper(root, causal_slice)
    if at_failure:
        roles.add(CausalRole.FAILURE_SITE)
    if root.cause_kind is RootCauseKind.IMPORT_RESOLUTION:
        roles.add(CausalRole.IMPORT_SOURCE)
    elif root.cause_kind is RootCauseKind.CONFIGURATION:
        roles.update({CausalRole.CONFIG_SOURCE, CausalRole.PRODUCER})
    elif node and node.kind in {CausalNodeKind.RETURN_VALUE, CausalNodeKind.LITERAL}:
        roles.update({CausalRole.RETURN_SOURCE, CausalRole.PRODUCER})
    elif node and node.kind is CausalNodeKind.PARAMETER:
        roles.update({CausalRole.ARGUMENT_SOURCE, CausalRole.CALLEE})
    elif node and node.kind is CausalNodeKind.CALL:
        roles.add(CausalRole.CALLER)
    else:
        roles.add(CausalRole.PRODUCER if not at_failure else CausalRole.CONSUMER)
    if transparent:
        roles.add(CausalRole.INTERMEDIATE_WRAPPER)

    if transparent:
        origin_role = OriginRole.INTERMEDIARY
        relation = RelationToFailure.INTERMEDIATE
    elif at_failure:
        origin_role = OriginRole.MANIFESTATION
        relation = RelationToFailure.MANIFESTATION
    elif node and node.kind is CausalNodeKind.PARAMETER:
        origin_role = OriginRole.BOUNDARY
        relation = RelationToFailure.UPSTREAM
    elif node:
        origin_role = OriginRole.SOURCE
        relation = RelationToFailure.UPSTREAM if root.causal_distance > 1 else RelationToFailure.DIRECT
    else:
        origin_role = OriginRole.UNKNOWN
        relation = RelationToFailure.UNKNOWN

    contract = {
        RootCauseKind.RETURN_CONTRACT: ContractRole.RETURN,
        RootCauseKind.ARGUMENT_BINDING: ContractRole.INPUT,
        RootCauseKind.CONFIGURATION: ContractRole.CONFIGURATION,
        RootCauseKind.IMPORT_RESOLUTION: ContractRole.IMPORT,
        RootCauseKind.ATTRIBUTE_VALUE: ContractRole.STATE,
        RootCauseKind.STATE_PROPAGATION: ContractRole.STATE,
    }.get(root.cause_kind, ContractRole.NONE)
    return RootCauseSignature(
        root.cause_kind,
        tuple(sorted(roles, key=lambda item: item.value)),
        origin_role,
        relation,
        contract,
    )


def compare_root_candidates(
    left: RootCauseCandidate,
    right: RootCauseCandidate,
    causal_slice: CausalSlice,
) -> RootPairwiseComparison:
    left_signature = root_cause_signature(left, causal_slice)
    right_signature = root_cause_signature(right, causal_slice)

    def result(preferred: RootCauseCandidate, rejected: RootCauseCandidate, reason: PairwiseRootReason) -> RootPairwiseComparison:
        return RootPairwiseComparison(preferred.id, rejected.id, reason, True)

    left_wrapper = CausalRole.INTERMEDIATE_WRAPPER in left_signature.causal_roles
    right_wrapper = CausalRole.INTERMEDIATE_WRAPPER in right_signature.causal_roles
    if left_wrapper != right_wrapper:
        return result(right, left, PairwiseRootReason.INTERMEDIATE_WRAPPER) if left_wrapper else result(left, right, PairwiseRootReason.INTERMEDIATE_WRAPPER)
    left_manifest = left_signature.origin_role is OriginRole.MANIFESTATION
    right_manifest = right_signature.origin_role is OriginRole.MANIFESTATION
    demonstrable_contract_origins = {
        RootCauseKind.NULL_FLOW,
        RootCauseKind.TYPE_FLOW,
    }
    if left_manifest != right_manifest:
        upstream = right if left_manifest else left
        symptom = left if left_manifest else right
        upstream_signature = right_signature if left_manifest else left_signature
        if upstream.cause_kind in demonstrable_contract_origins:
            return result(upstream, symptom, PairwiseRootReason.UPSTREAM_CONTRACT_ORIGIN)
    direct_source_kinds = demonstrable_contract_origins
    if (left.cause_kind in direct_source_kinds) != (right.cause_kind in direct_source_kinds):
        source, alternate = (left, right) if left.cause_kind in direct_source_kinds else (right, left)
        if (
            not (source.origin_path.startswith("tests/") or "/test" in source.origin_path)
            and source.structural_support == 1.0
            and source.completeness == 1.0
        ):
            return result(source, alternate, PairwiseRootReason.DIRECT_CAUSAL_ORIGIN)
    support_delta = (
        left.direct_support + left.structural_support + left.completeness
        - right.direct_support - right.structural_support - right.completeness
    )
    if abs(support_delta) >= 0.75:
        preferred, rejected = (left, right) if support_delta > 0 else (right, left)
        return result(preferred, rejected, PairwiseRootReason.STRONGER_STRUCTURAL_PATH)
    return RootPairwiseComparison(None, None, PairwiseRootReason.INSUFFICIENT_TO_DISTINGUISH, True)


def validate_root_comparison(
    preferred: RootCauseCandidate,
    rejected: RootCauseCandidate,
    reason: PairwiseRootReason,
    causal_slice: CausalSlice,
) -> bool:
    comparison = compare_root_candidates(preferred, rejected, causal_slice)
    if reason is PairwiseRootReason.INSUFFICIENT_TO_DISTINGUISH:
        return comparison.preferred_id is None
    return (
        comparison.preferred_id == preferred.id
        and comparison.rejected_id == rejected.id
        and comparison.reason is reason
    )


def semantic_dominant_root(
    candidates: Sequence[RootCauseCandidate], causal_slice: CausalSlice | None
) -> RootCauseCandidate | None:
    if not causal_slice or len(candidates) < 2:
        return candidates[0] if len(candidates) == 1 else None
    production = tuple(
        item for item in candidates
        if not (item.origin_path.startswith("tests/") or "/test" in item.origin_path)
    ) or tuple(candidates)
    winners: list[RootCauseCandidate] = []
    for candidate in production:
        comparisons = [
            compare_root_candidates(candidate, other, causal_slice)
            for other in production if other.id != candidate.id
        ]
        if comparisons and all(item.preferred_id == candidate.id for item in comparisons):
            winners.append(candidate)
    return winners[0] if len(winners) == 1 else None


def _node_for_target(target: RepairTarget, causal_slice: CausalSlice) -> CausalNode | None:
    if target.expression_id:
        found = next((item for item in causal_slice.nodes if item.id == target.expression_id), None)
        if found:
            return found
    return next((
        item for item in causal_slice.nodes
        if item.path == target.path
        and (target.line is None or item.line == target.line)
        and (
            not (target.symbol or target.parameter or target.attribute)
            or (item.symbol or "").rsplit(".", 1)[-1]
            in {
                (target.symbol or "").rsplit(".", 1)[-1],
                (target.parameter or "").rsplit(".", 1)[-1],
                (target.attribute or "").rsplit(".", 1)[-1],
            }
        )
    ), None)


def repair_target_signature(
    target: RepairTarget,
    strategy: RepairStrategyKind,
    root: RootCauseCandidate,
    causal_slice: CausalSlice,
) -> RepairTargetSignature:
    node = _node_for_target(target, causal_slice)
    root_signature = root_cause_signature(root, causal_slice)
    roles: set[CausalRole] = set()
    if target.path == root.origin_path:
        roles.update(root_signature.causal_roles)
    if target.path == root.failure_path:
        roles.add(CausalRole.FAILURE_SITE)
    if target.scope_kind is RepairTargetKind.CALL_SITE:
        roles.add(CausalRole.CALLER)
    elif target.scope_kind is RepairTargetKind.PARAMETER:
        roles.update({CausalRole.CALLEE, CausalRole.ARGUMENT_SOURCE})
    elif target.scope_kind is RepairTargetKind.RETURN_SITE:
        roles.update({CausalRole.PRODUCER, CausalRole.RETURN_SOURCE})
    elif target.scope_kind is RepairTargetKind.CONFIG_VALUE:
        roles.add(CausalRole.CONFIG_SOURCE)
    elif target.scope_kind is RepairTargetKind.IMPORT_EDGE:
        roles.add(CausalRole.IMPORT_SOURCE)

    relation_to_root = (
        TargetRelation.ROOT
        if node and root.causal_path and node.id == root.causal_path[0]
        else TargetRelation.ROOT_SCOPE
        if target.path == root.origin_path
        else TargetRelation.CAUSAL_PATH
        if node and node.id in root.causal_path
        else TargetRelation.UNRELATED
    )
    relation_to_failure = (
        TargetRelation.FAILURE
        if target.path == root.failure_path and target.line == root.failure_line
        else TargetRelation.CAUSAL_PATH
        if node and node.id in root.causal_path
        else TargetRelation.UNRELATED
    )
    distance = root.causal_path.index(node.id) if node and node.id in root.causal_path else None
    return RepairTargetSignature(
        target.scope_kind,
        tuple(sorted(roles, key=lambda item: item.value)),
        root_signature.origin_role,
        relation_to_root,
        relation_to_failure,
        distance,
    )


def target_preference(
    strategy: RepairStrategyKind,
    root: RootCauseCandidate,
    target: RepairTarget,
    causal_slice: CausalSlice,
) -> TargetPreference:
    signature = repair_target_signature(target, strategy, root, causal_slice)
    root_signature = root_cause_signature(root, causal_slice)
    preferred = {
        RepairStrategyKind.CORRECT_RETURN_VALUE: {RepairTargetKind.RETURN_SITE},
        RepairStrategyKind.FIX_PRODUCER: {RepairTargetKind.RETURN_SITE, RepairTargetKind.FUNCTION, RepairTargetKind.METHOD, RepairTargetKind.ATTRIBUTE},
        RepairStrategyKind.VALIDATE_BOUNDARY: {RepairTargetKind.PARAMETER, RepairTargetKind.FUNCTION, RepairTargetKind.METHOD},
        RepairStrategyKind.CORRECT_CONFIGURATION: {RepairTargetKind.CONFIG_VALUE},
        RepairStrategyKind.CORRECT_IMPORT: {RepairTargetKind.IMPORT_EDGE},
        RepairStrategyKind.CORRECT_TEST_EXPECTATION: {RepairTargetKind.TEST_EXPECTATION},
    }.get(strategy, set())
    if strategy is RepairStrategyKind.CORRECT_ARGUMENT:
        nodes = {item.id: item for item in causal_slice.nodes}
        incoming_argument = any(
            edge.target_id == root.causal_path[0]
            and edge.kind is CausalEdgeKind.PASSED_AS_ARGUMENT
            and edge.deterministic
            and edge.source_id in nodes
            and not (
                nodes[edge.source_id].path.startswith("tests/")
                or "/test" in nodes[edge.source_id].path
            )
            for edge in causal_slice.edges
        ) if root.causal_path else False
        preferred = {
            RepairTargetKind.CALL_SITE if incoming_argument else RepairTargetKind.PARAMETER
        }
    if target.scope_kind in preferred and signature.relation_to_root in {
        TargetRelation.ROOT, TargetRelation.ROOT_SCOPE, TargetRelation.CAUSAL_PATH,
    }:
        return TargetPreference.PREFERRED
    if signature.relation_to_root is TargetRelation.UNRELATED:
        return TargetPreference.INVALID
    if (
        root_signature.origin_role is OriginRole.INTERMEDIARY
        and target.path == root.origin_path
    ):
        return TargetPreference.INVALID
    return TargetPreference.COMPATIBLE


def semantic_target_equivalent(
    left: RepairTargetSignature, right: RepairTargetSignature
) -> bool:
    if left == right:
        return True
    boundary = {RepairTargetKind.PARAMETER, RepairTargetKind.FUNCTION, RepairTargetKind.METHOD}
    root_scope = {TargetRelation.ROOT, TargetRelation.ROOT_SCOPE}
    return (
        left.target_kind in boundary
        and right.target_kind in boundary
        and left.owning_symbol_role == right.owning_symbol_role
        and left.relation_to_root in root_scope
        and right.relation_to_root in root_scope
    )


def rank_target_preference(value: TargetPreference) -> float:
    return {
        TargetPreference.PREFERRED: 1.0,
        TargetPreference.COMPATIBLE: 0.65,
        TargetPreference.INVALID: 0.0,
    }[value]


def semantic_repair_strategy_score(
    strategy: RepairStrategyKind,
    root: RootCauseCandidate,
    causal_slice: CausalSlice,
    fallback: float,
) -> float:
    """Refine strategy quality only when a causal role resolves ownership."""
    if root.cause_kind is not RootCauseKind.ARGUMENT_BINDING or not root.causal_path:
        return fallback
    nodes = {item.id: item for item in causal_slice.nodes}
    has_production_argument_source = any(
        edge.target_id == root.causal_path[0]
        and edge.kind is CausalEdgeKind.PASSED_AS_ARGUMENT
        and edge.deterministic
        and edge.source_id in nodes
        and not (
            nodes[edge.source_id].path.startswith("tests/")
            or "/test" in nodes[edge.source_id].path
        )
        for edge in causal_slice.edges
    )
    if has_production_argument_source:
        return {
            RepairStrategyKind.CORRECT_ARGUMENT: 1.0,
            RepairStrategyKind.VALIDATE_BOUNDARY: 0.65,
        }.get(strategy, fallback)
    return {
        RepairStrategyKind.VALIDATE_BOUNDARY: 1.0,
        RepairStrategyKind.CORRECT_ARGUMENT: 0.70,
    }.get(strategy, fallback)


def pairwise_root_matrix(
    candidates: Iterable[RootCauseCandidate], causal_slice: CausalSlice
) -> tuple[RootPairwiseComparison, ...]:
    ordered = tuple(candidates)
    return tuple(
        compare_root_candidates(left, right, causal_slice)
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    )
