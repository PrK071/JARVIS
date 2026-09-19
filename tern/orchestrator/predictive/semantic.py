from __future__ import annotations

import ast
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .causal import (
    CausalResponsibilityKind,
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
    IMPORT_CYCLE = "IMPORT_CYCLE"


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
    ACTUAL_ARGUMENT_IS_DEFECT_SOURCE = "ACTUAL_ARGUMENT_IS_DEFECT_SOURCE"
    BINDING_RELATION_VIOLATED = "BINDING_RELATION_VIOLATED"
    RETURN_PRODUCER_IS_DEFECT_SOURCE = "RETURN_PRODUCER_IS_DEFECT_SOURCE"
    RETURN_CONTRACT_VIOLATED = "RETURN_CONTRACT_VIOLATED"
    MANIFESTATION_NOT_ORIGIN = "MANIFESTATION_NOT_ORIGIN"
    INSUFFICIENT_CONTRACT_EVIDENCE = "INSUFFICIENT_CONTRACT_EVIDENCE"
    INSUFFICIENT_BINDING_EVIDENCE = "INSUFFICIENT_BINDING_EVIDENCE"
    IMPORT_SCC_IS_CAUSAL_ENTITY = "IMPORT_SCC_IS_CAUSAL_ENTITY"


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
    entity_identity: str | None = None
    responsibility_kind: CausalResponsibilityKind = CausalResponsibilityKind.UNKNOWN
    defect_bearing_relation: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "cause_kind": self.cause_kind.value,
            "causal_roles": [item.value for item in self.causal_roles],
            "origin_role": self.origin_role.value,
            "relation_to_failure": self.relation_to_failure.value,
            "contract_role": self.contract_role.value,
            "entity_identity": self.entity_identity,
            "responsibility_kind": self.responsibility_kind.value,
            "defect_bearing_relation": self.defect_bearing_relation,
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


def _transparent_forwarded_names(expression: str) -> tuple[str, ...] | None:
    value = expression.strip()
    if not value.startswith("return "):
        return None
    returned = value[7:].strip()
    if not returned:
        return None
    try:
        parsed = ast.parse(value).body[0]
    except (SyntaxError, IndexError):
        return None
    if not isinstance(parsed, ast.Return) or parsed.value is None:
        return None
    if isinstance(parsed.value, (ast.Name, ast.Attribute)):
        name = parsed.value.id if isinstance(parsed.value, ast.Name) else None
        return (name,) if name else ()
    if not isinstance(parsed.value, ast.Call):
        return None
    forwarded = (*parsed.value.args, *(item.value for item in parsed.value.keywords))
    if not all(isinstance(item, (ast.Name, ast.Attribute)) for item in forwarded):
        return None
    names: list[str] = []
    for item in forwarded:
        current = item
        while isinstance(current, ast.Attribute):
            current = current.value
        if isinstance(current, ast.Name):
            names.append(current.id)
    return tuple(names)


def is_transparent_wrapper(root: RootCauseCandidate, causal_slice: CausalSlice) -> bool:
    node = _origin_node(root, causal_slice)
    if node is None or node.kind is not CausalNodeKind.RETURN_VALUE:
        return False
    forwarded_names = _transparent_forwarded_names(node.expression)
    if forwarded_names is None:
        return False
    try:
        returned = ast.parse(node.expression).body[0]
        returns_call = isinstance(returned, ast.Return) and isinstance(
            returned.value, ast.Call
        )
    except (SyntaxError, IndexError):
        return False
    nodes = {item.id: item for item in causal_slice.nodes}
    parameters = {
        item.symbol
        for item in nodes.values()
        if item.path == node.path
        and item.scope == node.scope
        and item.kind is CausalNodeKind.PARAMETER
        and item.symbol
    }
    # A forwarding wrapper passes its own inputs unchanged.  A call fed by a
    # local value (for example ``merged = ...; return parse(merged)``) is a
    # transformation boundary and may itself violate the return contract.
    incoming = _incoming(causal_slice)
    for name in forwarded_names:
        if name in parameters:
            # ``return parameter`` may own a broken return contract.  Only a
            # direct delegated call can transparently forward parameters.
            if not returns_call:
                return False
            continue
        local_nodes = [
            item for item in nodes.values()
            if item.path == node.path
            and item.scope == node.scope
            and item.kind is CausalNodeKind.LOCAL_VARIABLE
            and item.symbol == name
        ]
        if len(local_nodes) != 1:
            return False
        producers = [
            edge for edge in incoming.get(local_nodes[0].id, ())
            if edge.deterministic
            and edge.kind is CausalEdgeKind.RETURNED_FROM
            and nodes[edge.source_id].kind is CausalNodeKind.CALL
        ]
        if len(producers) != 1:
            return False
        call_inputs = [
            edge for edge in incoming.get(producers[0].source_id, ())
            if edge.deterministic
            and edge.kind is CausalEdgeKind.PASSED_AS_ARGUMENT
        ]
        if any(
            nodes[edge.source_id].kind is not CausalNodeKind.PARAMETER
            or nodes[edge.source_id].path != node.path
            or nodes[edge.source_id].scope != node.scope
            for edge in call_inputs
        ):
            return False
    queue: deque[tuple[str, int]] = deque([(node.id, 0)])
    visited = {node.id}
    allowed = {
        CausalEdgeKind.ASSIGNED_FROM,
        CausalEdgeKind.RETURNED_FROM,
        CausalEdgeKind.CALLS,
        CausalEdgeKind.PASSED_AS_ARGUMENT,
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
        if root.import_scc:
            roles.add(CausalRole.IMPORT_CYCLE)
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

    responsibility = (
        root.responsibility.kind
        if root.responsibility else CausalResponsibilityKind.UNKNOWN
    )
    demonstrated_return = bool(
        at_failure
        and root.cause_kind is RootCauseKind.RETURN_CONTRACT
        and root.contract_demonstrated
        and not transparent
    )
    if responsibility in {
        CausalResponsibilityKind.ARGUMENT_BINDING_DEFECT,
        CausalResponsibilityKind.CONSUMER_CONTRACT_DEFECT,
    }:
        origin_role = OriginRole.BOUNDARY
        relation = RelationToFailure.DIRECT if at_failure else RelationToFailure.UPSTREAM
    elif responsibility is CausalResponsibilityKind.ARGUMENT_SOURCE_DEFECT:
        origin_role = OriginRole.SOURCE
        relation = RelationToFailure.DIRECT if at_failure else RelationToFailure.UPSTREAM
    elif transparent:
        origin_role = OriginRole.INTERMEDIARY
        relation = RelationToFailure.INTERMEDIATE
    elif at_failure and not demonstrated_return:
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
        (
            f"IMPORT_SCC:{len(root.import_scc.member_modules)}:"
            f"{len(root.import_scc.production_edges)}"
            if root.import_scc else
            f"ARGUMENT_SLOT:{root.responsibility.parameter_ordinal}:"
            f"{'KEYWORD' if root.responsibility.keyword_binding else 'POSITIONAL'}"
            if root.responsibility and root.responsibility.binding_id else
            f"{root.cause_kind.value}:{origin_role.value}:{contract.value}"
        ),
        root.responsibility.kind
        if root.responsibility else CausalResponsibilityKind.UNKNOWN,
        root.responsibility.defect_bearing_relation
        if root.responsibility else None,
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

    left_scc = left.import_scc is not None
    right_scc = right.import_scc is not None
    if left_scc != right_scc and (
        left.cause_kind is RootCauseKind.IMPORT_RESOLUTION
        and right.cause_kind is RootCauseKind.IMPORT_RESOLUTION
    ):
        return result(
            left if left_scc else right,
            right if left_scc else left,
            PairwiseRootReason.IMPORT_SCC_IS_CAUSAL_ENTITY,
        )

    argument_responsibilities = {
        CausalResponsibilityKind.ARGUMENT_SOURCE_DEFECT,
        CausalResponsibilityKind.ARGUMENT_BINDING_DEFECT,
        CausalResponsibilityKind.CONSUMER_CONTRACT_DEFECT,
    }
    left_responsibility = (
        left.responsibility.kind
        if left.responsibility else CausalResponsibilityKind.UNKNOWN
    )
    right_responsibility = (
        right.responsibility.kind
        if right.responsibility else CausalResponsibilityKind.UNKNOWN
    )
    left_argument = left_responsibility in argument_responsibilities
    right_argument = right_responsibility in argument_responsibilities
    left_return = left_responsibility is CausalResponsibilityKind.RETURN_CONTRACT_DEFECT
    right_return = right_responsibility is CausalResponsibilityKind.RETURN_CONTRACT_DEFECT
    if (left_argument and right_return) or (right_argument and left_return):
        argument = left if left_argument else right
        returned = right if left_argument else left
        argument_profile = argument.responsibility
        returned_profile = returned.responsibility
        causally_connected = bool(
            argument_profile
            and returned_profile
            and (
                argument_profile.actual_argument_id in returned.causal_path
                or returned_profile.return_site_id in argument.causal_path
            )
        )
        if not causally_connected:
            return RootPairwiseComparison(
                None,
                None,
                PairwiseRootReason.INSUFFICIENT_TO_DISTINGUISH,
                True,
            )
        return_proven = bool(
            (
                returned.contract_demonstrated
                or returned.cause_kind in {
                    RootCauseKind.NULL_FLOW,
                    RootCauseKind.TYPE_FLOW,
                }
            )
            and returned.responsibility
            and returned.responsibility.return_site_id
        )
        binding_proven = bool(
            argument.responsibility and argument.responsibility.binding_id
        )
        if return_proven:
            return result(
                returned,
                argument,
                PairwiseRootReason.RETURN_CONTRACT_VIOLATED,
            )
        if binding_proven:
            reason = (
                PairwiseRootReason.ACTUAL_ARGUMENT_IS_DEFECT_SOURCE
                if argument.responsibility
                and argument.responsibility.kind
                is CausalResponsibilityKind.ARGUMENT_SOURCE_DEFECT
                else PairwiseRootReason.BINDING_RELATION_VIOLATED
            )
            return result(argument, returned, reason)
        return RootPairwiseComparison(
            None,
            None,
            PairwiseRootReason.INSUFFICIENT_TO_DISTINGUISH,
            True,
        )

    left_wrapper = CausalRole.INTERMEDIATE_WRAPPER in left_signature.causal_roles
    right_wrapper = CausalRole.INTERMEDIATE_WRAPPER in right_signature.causal_roles
    if left_wrapper != right_wrapper:
        return result(right, left, PairwiseRootReason.INTERMEDIATE_WRAPPER) if left_wrapper else result(left, right, PairwiseRootReason.INTERMEDIATE_WRAPPER)
    if (
        left.cause_kind is RootCauseKind.ARGUMENT_BINDING
        and right.cause_kind is RootCauseKind.ARGUMENT_BINDING
        and left.causal_path
        and right.causal_path
    ):
        left_binding = (
            left.responsibility.binding_id if left.responsibility else None
        )
        right_binding = (
            right.responsibility.binding_id if right.responsibility else None
        )
        same_formal_parameter = bool(
            left.responsibility
            and right.responsibility
            and left.responsibility.formal_parameter_id
            and left.responsibility.formal_parameter_id
            == right.responsibility.formal_parameter_id
        )
        if (
            left_binding
            and right_binding
            and left_binding != right_binding
            and same_formal_parameter
        ):
            return RootPairwiseComparison(
                None,
                None,
                PairwiseRootReason.INSUFFICIENT_TO_DISTINGUISH,
                True,
            )
        left_origin, right_origin = left.causal_path[0], right.causal_path[0]
        if right_origin in left.causal_path[1:]:
            return result(left, right, PairwiseRootReason.UPSTREAM_CONTRACT_ORIGIN)
        if left_origin in right.causal_path[1:]:
            return result(right, left, PairwiseRootReason.UPSTREAM_CONTRACT_ORIGIN)
    left_manifest = left_signature.origin_role is OriginRole.MANIFESTATION
    right_manifest = right_signature.origin_role is OriginRole.MANIFESTATION
    demonstrable_contract_origins = {
        RootCauseKind.NULL_FLOW,
        RootCauseKind.TYPE_FLOW,
        RootCauseKind.ARGUMENT_BINDING,
        RootCauseKind.RETURN_CONTRACT,
    }
    if left_manifest != right_manifest:
        upstream = right if left_manifest else left
        symptom = left if left_manifest else right
        upstream_signature = right_signature if left_manifest else left_signature
        if upstream.cause_kind in demonstrable_contract_origins:
            return result(upstream, symptom, PairwiseRootReason.UPSTREAM_CONTRACT_ORIGIN)
    left_config = CausalRole.CONFIG_SOURCE in left_signature.causal_roles
    right_config = CausalRole.CONFIG_SOURCE in right_signature.causal_roles
    if left_config != right_config:
        source, alternate = (left, right) if left_config else (right, left)
        alternate_signature = right_signature if left_config else left_signature
        if (
            source.structural_support == 1.0
            and source.completeness == 1.0
            and not (
                alternate.cause_kind is RootCauseKind.RETURN_CONTRACT
                and alternate_signature.contract_role is ContractRole.RETURN
                and alternate.contract_demonstrated
            )
        ):
            return result(
                source, alternate, PairwiseRootReason.UPSTREAM_CONTRACT_ORIGIN
            )
    left_return = (
        left.cause_kind is RootCauseKind.RETURN_CONTRACT
        and CausalRole.RETURN_SOURCE in left_signature.causal_roles
        and CausalRole.INTERMEDIATE_WRAPPER not in left_signature.causal_roles
    )
    right_return = (
        right.cause_kind is RootCauseKind.RETURN_CONTRACT
        and CausalRole.RETURN_SOURCE in right_signature.causal_roles
        and CausalRole.INTERMEDIATE_WRAPPER not in right_signature.causal_roles
    )
    if left_return != right_return:
        source, alternate = (left, right) if left_return else (right, left)
        alternate_signature = right_signature if left_return else left_signature
        if (
            source.structural_support == source.completeness == 1.0
            and (
                source.score >= 0.80
                and source.score - alternate.score >= 0.05
                or (
                    source.contract_demonstrated
                    and alternate_signature.origin_role
                    in {OriginRole.BOUNDARY, OriginRole.MANIFESTATION}
                )
                or (
                    source.contract_demonstrated
                    and CausalRole.CONFIG_SOURCE in alternate_signature.causal_roles
                )
            )
        ):
            return result(
                source, alternate, PairwiseRootReason.UPSTREAM_CONTRACT_ORIGIN
            )
    if left_return and right_return:
        stronger, weaker = (
            (left, right) if left.score > right.score else (right, left)
        )
        if (
            stronger.contract_demonstrated
            and not weaker.contract_demonstrated
        ) or (
            stronger.score >= 0.90 and stronger.score - weaker.score >= 0.08
        ):
            return result(
                stronger, weaker, PairwiseRootReason.STRONGER_STRUCTURAL_PATH
            )
    direct_source_kinds = {RootCauseKind.NULL_FLOW, RootCauseKind.TYPE_FLOW}
    if (left.cause_kind in direct_source_kinds) != (right.cause_kind in direct_source_kinds):
        source, alternate = (left, right) if left.cause_kind in direct_source_kinds else (right, left)
        source_binding = (
            source.responsibility.binding_id if source.responsibility else None
        )
        alternate_binding = (
            alternate.responsibility.binding_id
            if alternate.responsibility else None
        )
        same_binding = bool(
            source_binding
            and alternate_binding
            and source_binding == alternate_binding
        )
        structurally_connected = bool(
            alternate.causal_path
            and source.causal_path
            and (
                alternate.causal_path[0] in source.causal_path
                or source.causal_path[0] in alternate.causal_path
            )
        )
        slice_nodes = {item.id: item for item in causal_slice.nodes}
        actual_argument = (
            slice_nodes.get(alternate.responsibility.actual_argument_id)
            if alternate.responsibility else None
        )
        source_boundary = next((
            slice_nodes.get(node_id)
            for node_id in source.causal_path[1:]
            if slice_nodes.get(node_id)
            and slice_nodes[node_id].kind is CausalNodeKind.PARAMETER
        ), None)
        constructor_default_relation = bool(
            actual_argument
            and actual_argument.kind is CausalNodeKind.CALL
            and source_boundary
            and source_boundary.scope
            and source_boundary.scope.rsplit(".", 1)[0].rsplit(".", 1)[-1]
            == str(actual_argument.symbol or "").rsplit(".", 1)[-1]
        )
        unrelated_argument_bindings = bool(
            alternate.cause_kind is RootCauseKind.ARGUMENT_BINDING
            and not (
                same_binding
                or structurally_connected
                or constructor_default_relation
            )
        )
        if (
            not unrelated_argument_bindings
            and
            (source.origin_path, source.origin_line)
            != (alternate.origin_path, alternate.origin_line)
            and
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
    rejected: set[str] = set()
    # NULL/TYPE/RETURN candidates at one producer return site are alternate
    # representations of the same structural origin. Compare the strongest
    # representative with other causal roles instead of creating an
    # artificial tie between its representations.
    flow_family = {
        RootCauseKind.NULL_FLOW,
        RootCauseKind.TYPE_FLOW,
        RootCauseKind.RETURN_CONTRACT,
    }
    representation_priority = {
        RootCauseKind.NULL_FLOW: 0,
        RootCauseKind.TYPE_FLOW: 1,
        RootCauseKind.RETURN_CONTRACT: 2,
    }
    groups: dict[tuple[str, int], list[RootCauseCandidate]] = {}
    for item in production:
        signature = root_cause_signature(item, causal_slice)
        if (
            item.cause_kind in flow_family
            and {CausalRole.PRODUCER, CausalRole.RETURN_SOURCE}
            <= set(signature.causal_roles)
        ):
            groups.setdefault((item.origin_path, item.origin_line), []).append(item)
    grouped_representations = 0
    for values in groups.values():
        if len(values) < 2:
            continue
        representative = sorted(
            values,
            key=lambda item: (
                representation_priority[item.cause_kind],
                -item.score,
                item.id,
            ),
        )[0]
        duplicates = {item.id for item in values if item.id != representative.id}
        rejected.update(duplicates)
        grouped_representations += len(duplicates)

    # A literal default and its parameter node are two AST views of one
    # origin.  Prefer the concrete value-flow representation only when the
    # deterministic path directly binds that literal to that parameter.
    nodes = {item.id: item for item in causal_slice.nodes}
    for literal in production:
        if literal.cause_kind not in {RootCauseKind.NULL_FLOW, RootCauseKind.TYPE_FLOW}:
            continue
        if not literal.causal_path:
            continue
        origin = nodes.get(literal.causal_path[0])
        if origin is None or origin.kind is not CausalNodeKind.LITERAL:
            continue
        for binding in production:
            if (
                binding.cause_kind is RootCauseKind.ARGUMENT_BINDING
                and binding.origin_path == literal.origin_path
                and binding.origin_line == literal.origin_line
                and binding.causal_path
                and len(literal.causal_path) > 1
                and binding.causal_path[0] in literal.causal_path[1:]
            ):
                if binding.id not in rejected:
                    rejected.add(binding.id)
                    grouped_representations += 1
    decisive = grouped_representations
    for index, left in enumerate(production):
        for right in production[index + 1 :]:
            comparison = compare_root_candidates(left, right, causal_slice)
            if comparison.preferred_id and comparison.rejected_id:
                decisive += 1
                rejected.add(comparison.rejected_id)
    unbeaten = [item for item in production if item.id not in rejected]
    if decisive and len(unbeaten) == 1:
        return unbeaten[0]
    return None


def equivalent_root_origin(
    candidates: Sequence[RootCauseCandidate], causal_slice: CausalSlice | None
) -> RootCauseCandidate | None:
    """Choose a representation only when all unbeaten roots share one origin."""
    if not causal_slice:
        return None
    production = tuple(
        item for item in candidates
        if not (item.origin_path.startswith("tests/") or "/test" in item.origin_path)
    ) or tuple(candidates)
    rejected = {
        comparison.rejected_id
        for comparison in pairwise_root_matrix(production, causal_slice)
        if comparison.preferred_id and comparison.rejected_id
    }
    unbeaten = [item for item in production if item.id not in rejected]
    locations = {(item.origin_path, item.origin_line) for item in unbeaten}
    family = {
        RootCauseKind.NULL_FLOW,
        RootCauseKind.TYPE_FLOW,
        RootCauseKind.RETURN_CONTRACT,
    }
    signatures = [root_cause_signature(item, causal_slice) for item in unbeaten]
    if (
        unbeaten
        and len(locations) == 1
        and all(item.cause_kind in family for item in unbeaten)
        and all(
            {CausalRole.PRODUCER, CausalRole.RETURN_SOURCE}
            <= set(signature.causal_roles)
            for signature in signatures
        )
    ):
        return sorted(unbeaten, key=lambda item: (-item.score, item.id))[0]
    return None


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
    origin = _origin_node(root, causal_slice)
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

    target_owner = (target.symbol or "").rsplit(".", 1)[-1]
    root_owner = (
        (origin.scope if origin and origin.scope != "module" else None)
        or root.origin_symbol
        or ""
    ).rsplit(".", 1)[-1]
    same_owner_scope = bool(target_owner and root_owner and target_owner == root_owner)
    binding_target = bool(
        root.responsibility
        and root.responsibility.binding_id
        and target.expression_id == root.responsibility.producer_id
    )
    relation_to_root = (
        TargetRelation.ROOT
        if node and root.causal_path and node.id == root.causal_path[0]
        else TargetRelation.ROOT_SCOPE
        if target.path == root.origin_path and same_owner_scope
        else TargetRelation.CAUSAL_PATH
        if binding_target or node and node.id in root.causal_path
        else TargetRelation.UNRELATED
    )
    relation_to_failure = (
        TargetRelation.FAILURE
        if target.path == root.failure_path and target.line == root.failure_line
        else TargetRelation.CAUSAL_PATH
        if node and node.id in root.causal_path
        else TargetRelation.UNRELATED
    )
    distance = (
        root.causal_path.index(node.id)
        if node and node.id in root.causal_path else 0 if binding_target else None
    )
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
        responsibility = (
            root.responsibility.kind
            if root.responsibility else CausalResponsibilityKind.UNKNOWN
        )
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
        forwarded_argument = any(
            edge.source_id == root.causal_path[0]
            and edge.target_id in root.causal_path[1:]
            and edge.kind is CausalEdgeKind.PASSED_AS_ARGUMENT
            and edge.deterministic
            and nodes.get(edge.target_id) is not None
            and nodes[edge.target_id].kind is CausalNodeKind.PARAMETER
            for edge in causal_slice.edges
        ) if root.causal_path else False
        preferred = (
            {RepairTargetKind.CALL_SITE}
            if responsibility is CausalResponsibilityKind.ARGUMENT_SOURCE_DEFECT
            else {RepairTargetKind.PARAMETER}
            if (
                responsibility is CausalResponsibilityKind.ARGUMENT_BINDING_DEFECT
                and root.responsibility is not None
                and root.responsibility.defect_bearing_relation
                == "default_argument_to_formal_parameter"
            )
            else {RepairTargetKind.CALL_SITE}
            if responsibility is CausalResponsibilityKind.ARGUMENT_BINDING_DEFECT
            else {RepairTargetKind.PARAMETER}
            if responsibility is CausalResponsibilityKind.CONSUMER_CONTRACT_DEFECT
            else
            {RepairTargetKind.CALL_SITE, RepairTargetKind.PARAMETER}
            if forwarded_argument
            else {RepairTargetKind.CALL_SITE}
            if incoming_argument
            else {RepairTargetKind.PARAMETER}
        )
    preferred_relations = {TargetRelation.ROOT, TargetRelation.ROOT_SCOPE}
    if (
        strategy is RepairStrategyKind.CORRECT_ARGUMENT
        and target.scope_kind is RepairTargetKind.CALL_SITE
    ):
        preferred_relations.add(TargetRelation.CAUSAL_PATH)
    test_only_call_site = (
        strategy is RepairStrategyKind.CORRECT_ARGUMENT
        and target.scope_kind is RepairTargetKind.CALL_SITE
        and (target.path.startswith("tests/") or "/test" in target.path)
    )
    if (
        target.scope_kind in preferred
        and signature.relation_to_root in preferred_relations
        and not test_only_call_site
    ):
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
    signature = root_cause_signature(root, causal_slice)
    producer_return = {
        CausalRole.PRODUCER,
        CausalRole.RETURN_SOURCE,
    } <= set(signature.causal_roles)
    if producer_return and root.cause_kind is RootCauseKind.NULL_FLOW:
        return {
            RepairStrategyKind.FIX_PRODUCER: 1.0,
            RepairStrategyKind.CORRECT_RETURN_VALUE: 0.92,
            RepairStrategyKind.VALIDATE_BOUNDARY: 0.65,
        }.get(strategy, fallback)
    if producer_return and root.cause_kind is RootCauseKind.TYPE_FLOW:
        return {
            RepairStrategyKind.CORRECT_RETURN_VALUE: 1.0,
            RepairStrategyKind.FIX_PRODUCER: 0.92,
            RepairStrategyKind.FIX_CONSUMER_CONTRACT: 0.60,
            RepairStrategyKind.VALIDATE_BOUNDARY: 0.55,
            RepairStrategyKind.CORRECT_ARGUMENT: 0.40,
        }.get(strategy, fallback)
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
    forwards_to_parameter = any(
        edge.source_id == root.causal_path[0]
        and edge.target_id in root.causal_path[1:]
        and edge.kind is CausalEdgeKind.PASSED_AS_ARGUMENT
        and edge.deterministic
        and nodes.get(edge.target_id) is not None
        and nodes[edge.target_id].kind is CausalNodeKind.PARAMETER
        for edge in causal_slice.edges
    )
    if has_production_argument_source or forwards_to_parameter:
        return {
            RepairStrategyKind.CORRECT_ARGUMENT: 1.0,
            RepairStrategyKind.VALIDATE_BOUNDARY: 0.65,
        }.get(strategy, fallback)
    return {
        RepairStrategyKind.VALIDATE_BOUNDARY: 1.0,
        RepairStrategyKind.CORRECT_ARGUMENT: 0.70,
    }.get(strategy, fallback)


def semantic_repair_pair_score(
    strategy: RepairStrategyKind,
    root: RootCauseCandidate,
    target: RepairTarget,
    causal_slice: CausalSlice,
    fallback: float,
) -> float:
    """Score strategy and target as one structural repair decision."""
    score = semantic_repair_strategy_score(
        strategy, root, causal_slice, fallback
    )
    if (
        root.cause_kind is RootCauseKind.NULL_FLOW
        and strategy is RepairStrategyKind.FIX_PRODUCER
        and target.scope_kind is RepairTargetKind.RETURN_SITE
    ):
        # At a return site this strategy is merely a less precise spelling of
        # CORRECT_RETURN_VALUE.  Reserve producer preference for a producer
        # scope (function/method/attribute), where ownership is explicit.
        return min(score, 0.90)
    return score


def pairwise_root_matrix(
    candidates: Iterable[RootCauseCandidate], causal_slice: CausalSlice
) -> tuple[RootPairwiseComparison, ...]:
    ordered = tuple(candidates)
    return tuple(
        compare_root_candidates(left, right, causal_slice)
        for index, left in enumerate(ordered)
        for right in ordered[index + 1 :]
    )
