from __future__ import annotations

import json
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..analysis import PredictiveAnalyzer, StructuredReasoner
from ..evaluation import _retrieval, evaluate_predictive_cases
from .generator import (
    GENERATOR_VERSION,
    CausalOracle,
    GeneratedProgram,
    ProgramFamily,
    ProgramSpec,
    generate_program,
    suite_sha256,
)


@dataclass(frozen=True)
class MetricValue:
    numerator: int
    denominator: int
    population: str

    @property
    def value(self) -> float | None:
        return self.numerator / self.denominator if self.denominator else None

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "population": self.population,
        }


def _metric(values: Iterable[bool], population: str) -> dict[str, Any]:
    items = tuple(values)
    return MetricValue(sum(items), len(items), population).as_dict()


def _root_matches(root: Mapping[str, Any], oracle: CausalOracle) -> bool:
    if oracle.abstention_expected:
        return False
    responsibility = (root.get("responsibility") or {}).get("kind")
    if responsibility not in oracle.responsibility_kinds:
        return False
    if root.get("cause_kind") not in oracle.root_kinds:
        return False
    if oracle.import_scc_members:
        members = set((root.get("import_scc") or {}).get("member_modules") or ())
        if members != set(oracle.import_scc_members):
            return False
    if oracle.binding_slot is not None and responsibility in {
        "ARGUMENT_SOURCE_DEFECT", "ARGUMENT_BINDING_DEFECT",
    }:
        profile = root.get("responsibility") or {}
        if profile.get("parameter_ordinal") != oracle.binding_slot:
            return False
        if oracle.binding_keyword is not None and profile.get("keyword_binding") != oracle.binding_keyword:
            return False
    if oracle.origin_paths and root.get("origin_path") not in oracle.origin_paths:
        # Argument roots may be represented at either side of a same-file binding.
        if not set(oracle.origin_paths) == {root.get("failure_path")}:
            return False
    return True


def _relation_matches(root: Mapping[str, Any], oracle: CausalOracle) -> bool:
    relation = str((root.get("responsibility") or {}).get("defect_bearing_relation") or "")
    aliases = {
        "ACTUAL_TO_FORMAL": "actual_argument_to_formal_parameter",
        "DEFAULT_TO_FORMAL": "default_argument_to_formal_parameter",
        "RETURN_TO_CONSUMER": "return_site_to_caller_assignment",
        "PRODUCTION_IMPORT_CYCLE": "import_edge_to_module_initialization",
    }
    expected = {aliases.get(item, item).casefold() for item in oracle.defect_bearing_relations}
    return relation.casefold() in expected


def _selected_root(actual: Mapping[str, Any]) -> str | None:
    for item in actual.get("hypotheses") or ():
        if item.get("root_cause_id"):
            return str(item["root_cause_id"])
    return None


def _recommended(actual: Mapping[str, Any]) -> Mapping[str, Any] | None:
    candidate_id = actual.get("recommended_candidate_id")
    return next(
        (item for item in actual.get("candidates") or () if item.get("id") == candidate_id),
        None,
    )


def structural_case_metrics(program: GeneratedProgram, retrieval: Mapping[str, Any]) -> dict[str, Any]:
    oracle = program.oracle
    roots = tuple(retrieval.get("root_cause_candidates") or ())
    matching = tuple(root for root in roots if _root_matches(root, oracle))
    if oracle.abstention_expected:
        candidate_recall = not matching
        relation_recall = True
    else:
        candidate_recall = bool(matching)
        relation_recall = any(_relation_matches(root, oracle) for root in matching)
    slot_recall: bool | None = None
    if oracle.binding_slot is not None:
        slot_recall = any(
            (root.get("responsibility") or {}).get("parameter_ordinal") == oracle.binding_slot
            for root in matching
        )
    scc_recall: bool | None = None
    if oracle.import_scc_members:
        scc_recall = any(
            set((root.get("import_scc") or {}).get("member_modules") or ())
            == set(oracle.import_scc_members)
            for root in roots
        )
    retrieved = set(retrieval.get("retrieved_files") or ())
    expected_files = set(oracle.origin_paths)
    retrieval_hit = not expected_files or bool(retrieved & expected_files)
    return {
        "retrieval_accuracy": retrieval_hit,
        "oracle_root_candidate_recall": candidate_recall,
        "oracle_relation_recall": relation_recall,
        "oracle_slot_identity_recall": slot_recall,
        "oracle_scc_recall": scc_recall,
        "root_candidates": len(roots),
    }


def live_case_metrics(program: GeneratedProgram, result: Mapping[str, Any]) -> dict[str, Any]:
    oracle = program.oracle
    actual = result.get("actual") or {}
    retrieval = result.get("retrieval") or {}
    roots = {
        item.get("id"): item
        for item in (actual.get("root_cause_candidates") or retrieval.get("root_cause_candidates") or ())
    }
    selected = roots.get(_selected_root(actual))
    candidate = _recommended(actual)
    recommended = candidate is not None
    selected_valid = bool(selected and _root_matches(selected, oracle))
    strategy_valid = bool(
        candidate and candidate.get("strategy_kind") in oracle.expected_repair_strategies
    )
    targets = candidate.get("repair_targets") or () if candidate else ()
    target_valid = bool(
        candidate and any(item.get("scope_kind") in oracle.valid_target_roles for item in targets)
    )
    correct_abstention = oracle.abstention_expected and not recommended
    return {
        "root_selection_given_oracle_candidate": (
            selected_valid if not oracle.abstention_expected else correct_abstention
        ),
        "repair_given_correct_root": strategy_valid if selected_valid else None,
        "target_given_correct_root": target_valid if selected_valid and strategy_valid else None,
        "root_validity": selected_valid if not oracle.abstention_expected else correct_abstention,
        "repair_pair_validity": (
            strategy_valid and target_valid if not oracle.abstention_expected else correct_abstention
        ),
        "top1_validity": (
            selected_valid and strategy_valid and target_valid
            if recommended else correct_abstention
        ),
        "recommendation_precision": (
            selected_valid and strategy_valid and target_valid if recommended else None
        ),
        "recommendation_coverage": recommended,
        "false_abstention": not oracle.abstention_expected and not recommended,
        "argument_binding_accuracy": (
            selected_valid if "ARGUMENT_SOURCE_DEFECT" in oracle.responsibility_kinds
            or "ARGUMENT_BINDING_DEFECT" in oracle.responsibility_kinds else None
        ),
        "return_contract_accuracy": (
            selected_valid if "RETURN_CONTRACT_DEFECT" in oracle.responsibility_kinds else None
        ),
        "argument_vs_return_accuracy": (
            selected_valid
            if program.spec.family is ProgramFamily.ARGUMENT_VS_RETURN
            and not oracle.abstention_expected
            else None
        ),
        "slot_identity_accuracy": (
            selected_valid if oracle.binding_slot is not None else None
        ),
        "import_scc_precision": (
            selected_valid if selected and selected.get("cause_kind") == "IMPORT_RESOLUTION" else None
        ),
        "import_scc_recall": (
            selected_valid if oracle.import_scc_members else None
        ),
        "observer_exclusion_accuracy": (
            selected_valid if oracle.import_scc_members and program.spec.observer_import else None
        ),
    }


def _summarize(rows: Sequence[Mapping[str, Any]], key: str, population: str) -> dict[str, Any]:
    values = [bool(row[key]) for row in rows if row.get(key) is not None]
    return _metric(values, population)


def _matrix(
    rows: Sequence[Mapping[str, Any]], row_key: str, column_key: str, metric: str
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        value = (row.get("metrics") or {}).get(metric)
        if value is None:
            continue
        spec = row["spec"]
        groups[(str(spec[row_key]), str(spec[column_key]))].append(bool(value))
    result: dict[str, dict[str, dict[str, Any]]] = {}
    for (row_name, column_name), values in sorted(groups.items()):
        result.setdefault(row_name, {})[column_name] = _metric(
            values, f"{row_key}={row_name},{column_key}={column_name}"
        )
    return result


def _failure_fingerprint(program: GeneratedProgram, metrics: Mapping[str, Any]) -> str | None:
    failed = [key for key, value in metrics.items() if value is False]
    if not failed:
        return None
    dimensions = ",".join(f"{key}={value}" for key, value in program.spec.fingerprint)
    return f"{'+'.join(sorted(failed))}|{dimensions}"


def evaluate_generated_structural(
    specs: Sequence[ProgramSpec], *, output_root: Path | None = None
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    fingerprints: Counter[str] = Counter()
    temp = tempfile.TemporaryDirectory(prefix="jarvis-predictive-v10-") if output_root is None else None
    root = output_root or Path(temp.name)
    try:
        for spec in sorted(specs, key=lambda item: item.id):
            fixture = root / spec.id
            program = generate_program(spec, fixture)
            _context, retrieval = _retrieval(program.case)
            metrics = structural_case_metrics(program, retrieval)
            fingerprint = _failure_fingerprint(program, metrics)
            if fingerprint:
                fingerprints[fingerprint] += 1
            rows.append({
                "id": spec.id,
                "family": spec.family.value,
                "split": spec.split,
                "population": program.oracle.population.value,
                "spec": json.loads(spec.canonical_json()),
                "oracle": program.oracle.as_dict(),
                "metrics": metrics,
                "failure_fingerprint": fingerprint,
                "retrieval": retrieval,
            })
    finally:
        if temp is not None:
            temp.cleanup()
    names = (
        "retrieval_accuracy", "oracle_root_candidate_recall", "oracle_relation_recall",
        "oracle_slot_identity_recall", "oracle_scc_recall",
    )
    metric_rows = [row["metrics"] for row in rows]
    metrics = {
        name: _summarize(metric_rows, name, "generated structural cases")
        for name in names
    }
    family: dict[str, Any] = {}
    for family_name in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family_name]
        family[family_name] = {
            name: _summarize(
                [row["metrics"] for row in family_rows], name, f"family={family_name}"
            )
            for name in names
        }
    return {
        "version": 10,
        "generator_version": GENERATOR_VERSION,
        "mode": "structural",
        "split": specs[0].split if specs and len({item.split for item in specs}) == 1 else "mixed",
        "cases": len(rows),
        "suite_sha256": suite_sha256(specs),
        "metrics": metrics,
        "family_metrics": family,
        "argument_matrix": _matrix(rows, "argument_style", "wrapper_depth", "oracle_root_candidate_recall"),
        "import_matrix": _matrix(rows, "scc_size", "observer_import", "oracle_scc_recall"),
        "failure_fingerprints": [
            {"fingerprint": name, "count": count}
            for name, count in fingerprints.most_common(10)
        ],
        "results": rows,
    }


def evaluate_generated_live(
    specs: Sequence[ProgramSpec],
    *,
    reasoner: StructuredReasoner,
    runs: int = 1,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    temporary_workspace = (
        tempfile.TemporaryDirectory(prefix="jarvis-predictive-v10-live-")
        if workspace_root is None
        else None
    )
    try:
        root = (
            Path(temporary_workspace.name)
            if temporary_workspace is not None
            else workspace_root
        )
        assert root is not None
        root.mkdir(parents=True, exist_ok=True)
        programs = [generate_program(spec, root / spec.id) for spec in specs]
        raw = evaluate_predictive_cases(
            [item.case for item in programs], mode="live", reasoner=reasoner, runs=runs
        )
        by_id = {program.spec.id: program for program in programs}
        rows = []
        for result in raw["results"]:
            program = by_id[result["id"]]
            metrics = live_case_metrics(program, result)
            rows.append(dict(result) | {
                "spec": json.loads(program.spec.canonical_json()),
                "oracle": program.oracle.as_dict(),
                "metrics_v10": metrics,
            })
    finally:
        if temporary_workspace is not None:
            temporary_workspace.cleanup()
    metric_names = (
        "root_selection_given_oracle_candidate", "repair_given_correct_root",
        "target_given_correct_root", "root_validity", "repair_pair_validity",
        "top1_validity", "recommendation_precision", "recommendation_coverage",
        "false_abstention", "argument_binding_accuracy", "return_contract_accuracy",
        "argument_vs_return_accuracy", "slot_identity_accuracy", "import_scc_precision",
        "import_scc_recall", "observer_exclusion_accuracy",
    )
    summary = {
        name: _summarize(
            [row["metrics_v10"] for row in rows], name, "generated live cases"
        ) for name in metric_names
    }
    family_metrics: dict[str, Any] = {}
    for family_name in sorted({row["spec"]["family"] for row in rows}):
        family_rows = [
            row["metrics_v10"] for row in rows
            if row["spec"]["family"] == family_name
        ]
        family_metrics[family_name] = {
            name: _summarize(family_rows, name, f"family={family_name}")
            for name in metric_names
        }
    return dict(raw) | {
        "version": 10,
        "generator_version": GENERATOR_VERSION,
        "suite_sha256": suite_sha256(specs),
        "layer_metrics": summary,
        "family_metrics_v10": family_metrics,
        "results": rows,
    }


def _selected_responsibility(result: Mapping[str, Any]) -> str | None:
    actual = result.get("actual") or {}
    roots = {
        item.get("id"): item
        for item in (
            actual.get("root_cause_candidates")
            or (result.get("retrieval") or {}).get("root_cause_candidates")
            or ()
        )
    }
    selected = roots.get(_selected_root(actual))
    return (selected.get("responsibility") or {}).get("kind") if selected else None


def _recommended_semantics(
    result: Mapping[str, Any],
) -> tuple[str | None, tuple[str, ...]]:
    candidate = _recommended(result.get("actual") or {})
    if candidate is None:
        return None, ()
    targets = tuple(sorted({
        str(item.get("scope_kind"))
        for item in candidate.get("repair_targets") or ()
        if item.get("scope_kind")
    }))
    return candidate.get("strategy_kind"), targets


def evaluate_generated_counterfactuals(
    pairs: Sequence[tuple[ProgramSpec, ProgramSpec]],
    *,
    reasoner: StructuredReasoner,
    runs: int = 1,
    workspace_root: Path | None = None,
) -> dict[str, Any]:
    """Evaluate causal sensitivity across paired, oracle-owned changes.

    Each side is first evaluated normally.  A pair counts as sensitive only
    when both decisions are correct *and* their semantic identity changes in
    the dimension changed by the generator.  This prevents an unstable pair
    of wrong answers from looking counterfactually sensitive.
    """
    specs = tuple(spec for pair in pairs for spec in pair)
    report = evaluate_generated_live(
        specs, reasoner=reasoner, runs=runs, workspace_root=workspace_root
    )
    by_id = {str(row["id"]): row for row in report["results"]}
    pair_rows: list[dict[str, Any]] = []
    for base_spec, changed_spec in pairs:
        base = by_id[base_spec.id]
        changed = by_id[changed_spec.id]
        base_root = _selected_responsibility(base)
        changed_root = _selected_responsibility(changed)
        base_strategy, base_targets = _recommended_semantics(base)
        changed_strategy, changed_targets = _recommended_semantics(changed)
        base_metrics = base["metrics_v10"]
        changed_metrics = changed["metrics_v10"]
        root_sensitive = bool(
            base_metrics["root_validity"]
            and changed_metrics["root_validity"]
            and base_root
            and changed_root
            and base_root != changed_root
        )
        repair_sensitive = bool(
            base_metrics["repair_pair_validity"]
            and changed_metrics["repair_pair_validity"]
            and base_strategy
            and changed_strategy
            and base_strategy != changed_strategy
        )
        target_sensitive = bool(
            base_metrics["repair_pair_validity"]
            and changed_metrics["repair_pair_validity"]
            and base_targets
            and changed_targets
            and base_targets != changed_targets
        )
        pair_rows.append({
            "base_id": base_spec.id,
            "counterfactual_id": changed_spec.id,
            "base_root_responsibility": base_root,
            "counterfactual_root_responsibility": changed_root,
            "base_strategy": base_strategy,
            "counterfactual_strategy": changed_strategy,
            "base_target_roles": list(base_targets),
            "counterfactual_target_roles": list(changed_targets),
            "root_sensitivity": root_sensitive,
            "repair_sensitivity": repair_sensitive,
            "target_sensitivity": target_sensitive,
        })
    counterfactual_metrics = {
        name: _summarize(pair_rows, name, "generated counterfactual pairs")
        for name in (
            "root_sensitivity", "repair_sensitivity", "target_sensitivity",
        )
    }
    return report | {
        "split": "counterfactual",
        "pairs": len(pair_rows),
        "counterfactual_metrics": counterfactual_metrics,
        "counterfactual_results": pair_rows,
    }


def evaluate_oracle_injection(
    program: GeneratedProgram,
    *, reasoner: StructuredReasoner,
    stage: str = "root_candidates",
) -> dict[str, Any]:
    if stage not in {"root_candidates", "root"}:
        raise ValueError("oracle stage must be root_candidates or root")
    context, retrieval = _retrieval(program.case)
    matching = tuple(
        root for root in context.root_cause_candidates
        if _root_matches(root.as_dict(), program.oracle)
    )
    if not matching:
        return {"stage": stage, "oracle_candidates": 0, "analysis": None}
    injected = matching if stage == "root_candidates" else matching[:1]
    analysis = PredictiveAnalyzer(reasoner).analyze_with_diagnostics(
        replace(context, root_cause_candidates=injected)
    )
    return {
        "stage": stage,
        "oracle_candidates": len(injected),
        "retrieval": retrieval,
        "analysis": {
            "failure_reason": analysis.failure_reason.value if analysis.failure_reason else None,
            "selected_roots": [item.selected_id for item in analysis.root_cause_selections],
            "strategies": [item.strategy_kind.value for item in analysis.repair_strategies],
            "candidates": [item.as_dict() for item in analysis.candidates],
        },
    }


def reduce_failing_spec(
    spec: ProgramSpec,
    predicate,
) -> ProgramSpec:
    """Greedily reduce structural dimensions while preserving a failure predicate."""
    current = spec
    reductions = (
        ("wrapper_depth", 0),
        ("slot_count", max(1, spec.defective_slot + 1)),
        ("temporary_variable", False),
        ("attribute_fed", False),
        ("return_fed", False),
        ("distractor", False),
        ("observer_import", False),
    )
    for field_name, value in reductions:
        candidate = replace(current, **{field_name: value})
        if predicate(candidate):
            current = candidate
    return current


def format_generated_evaluation(report: Mapping[str, Any]) -> str:
    def value(name: str) -> str:
        item = (report.get("metrics") or report.get("layer_metrics") or {}).get(name) or {}
        number = item.get("value")
        return "-" if number is None else f"{100 * float(number):.1f}% ({item['numerator']}/{item['denominator']})"

    lines = [
        "Predictive Causal Generalization v10",
        "",
        f"Mode: {report.get('mode', 'live')}",
        f"Split: {report.get('split')}",
        f"Cases: {report.get('cases')}",
        f"Generator: v{report.get('generator_version')}",
        f"Suite SHA256: {report.get('suite_sha256')}",
    ]
    for name in (
        "retrieval_accuracy", "oracle_root_candidate_recall",
        "oracle_relation_recall", "oracle_slot_identity_recall", "oracle_scc_recall",
        "root_validity", "repair_pair_validity", "top1_validity",
        "recommendation_precision", "false_abstention",
    ):
        metrics = report.get("metrics") or report.get("layer_metrics") or {}
        if name in metrics:
            lines.append(f"{name}: {value(name)}")
    for name, item in (report.get("counterfactual_metrics") or {}).items():
        number = item.get("value")
        rendered = "-" if number is None else (
            f"{100 * float(number):.1f}% ({item['numerator']}/{item['denominator']})"
        )
        lines.append(f"{name}: {rendered}")
    return "\n".join(lines)
