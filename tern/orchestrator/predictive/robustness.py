from __future__ import annotations

import ast
import copy
import hashlib
import io
import json
import re
import shutil
import tokenize
from collections import Counter
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Callable, Mapping, Sequence

from .analysis import PredictiveAnalyzer
from .benchmark_v4 import CausalTruth, EvaluationStatus
from .benchmark_v5 import (
    BenchmarkAdjudicationV5,
    RepairTruthV5,
    adjudicate_predictive_result_v5,
    load_benchmark_v5_adjudications,
)
from .evaluation import (
    CORPUS_ROOT,
    CountingReasoner,
    PredictiveCase,
    _retrieval,
    evaluate_predictive_cases,
    load_predictive_cases,
)
from .repair import RepairTarget
from .models import EvidenceLedger, ProblemContext


ROBUSTNESS_ROOT = CORPUS_ROOT / "v6"


class MetamorphicTransformation(str, Enum):
    SYMBOL_RENAME = "SYMBOL_RENAME"
    FILE_RENAME = "FILE_RENAME"
    HARMLESS_COMMENT = "HARMLESS_COMMENT"
    DOCSTRING_DISTRACTOR = "DOCSTRING_DISTRACTOR"
    ERROR_PARAPHRASE = "ERROR_PARAPHRASE"
    FUNCTION_REORDER = "FUNCTION_REORDER"
    TEMPORARY_VARIABLE = "TEMPORARY_VARIABLE"
    POSITIONAL_TO_KEYWORD = "POSITIONAL_TO_KEYWORD"
    LEXICAL_DISTRACTOR = "LEXICAL_DISTRACTOR"
    WRAPPER_FUNCTION = "WRAPPER_FUNCTION"


class FeatureProvenance(str, Enum):
    STRUCTURAL = "STRUCTURAL"
    TRACEBACK = "TRACEBACK"
    USER_TEXT = "USER_TEXT"
    LEXICAL = "LEXICAL"
    TEST = "TEST"
    CONFIG = "CONFIG"
    MODEL = "MODEL"


class AbstentionStage(str, Enum):
    RETRIEVAL = "RETRIEVAL"
    CAUSAL_SLICE = "CAUSAL_SLICE"
    ROOT_SELECTION = "ROOT_SELECTION"
    REPAIR_GENERATION = "REPAIR_GENERATION"
    TARGET_VALIDATION = "TARGET_VALIDATION"
    POLICY = "POLICY"
    RANKING = "RANKING"
    RECOMMENDED = "RECOMMENDED"


@dataclass(frozen=True)
class MetricDefinition:
    name: str
    numerator: str
    denominator: str
    eligible_cases: str

    def as_dict(self) -> dict[str, str]:
        return {
            "name": self.name,
            "numerator": self.numerator,
            "denominator": self.denominator,
            "eligible_cases": self.eligible_cases,
        }


@dataclass(frozen=True)
class MetamorphicExpectation:
    root_cause: bool = True
    repair_strategy: bool = True
    repair_target: bool = True
    recommendation: bool = True
    abstention: bool = True
    causal_path: bool = True


@dataclass(frozen=True)
class MetamorphicCase:
    id: str
    base_case_id: str
    split: str
    transformation: MetamorphicTransformation
    parameters: Mapping[str, Any]
    expectation: MetamorphicExpectation = MetamorphicExpectation()


@dataclass(frozen=True)
class CounterfactualCase:
    id: str
    base_case_id: str
    counterfactual_case_id: str
    split: str
    root_must_change: bool = True
    repair_must_change: bool = True
    target_must_change: bool = False


@dataclass(frozen=True)
class TransformationMap:
    paths: Mapping[str, str]
    symbols: Mapping[str, str]
    line_offsets: Mapping[str, int]

    @property
    def inverse_paths(self) -> dict[str, str]:
        return {value: key for key, value in self.paths.items()}

    @property
    def inverse_symbols(self) -> dict[str, str]:
        return {value: key for key, value in self.symbols.items()}


METRIC_DEFINITIONS = (
    MetricDefinition(
        "evidence_reference_validity",
        "emitted evidence references whose path and inclusive line range exist",
        "all emitted evidence references",
        "all evaluated cases",
    ),
    MetricDefinition(
        "unsupported_claim_rate",
        "claims classified UNSUPPORTED by deterministic ledger validation",
        "all emitted hypothesis claims",
        "cases that emitted claims",
    ),
    MetricDefinition(
        "root_cause_validity",
        "positive cases with at least one selected structurally acceptable root",
        "all evaluable non-abstention cases, including false abstentions",
        "CANONICAL or MULTIPLE_VALID_ANSWERS and abstention_expected=false",
    ),
    MetricDefinition(
        "repair_pair_validity",
        "positive cases generating at least one compatible strategy-target pair",
        "all evaluable non-abstention cases, including false abstentions",
        "CANONICAL or MULTIPLE_VALID_ANSWERS and abstention_expected=false",
    ),
    MetricDefinition(
        "repair_strategy_validity",
        "positive cases generating at least one acceptable repair strategy",
        "all evaluable non-abstention cases, including false abstentions",
        "CANONICAL or MULTIPLE_VALID_ANSWERS and abstention_expected=false",
    ),
    MetricDefinition(
        "repair_target_validity",
        "positive cases generating at least one acceptable repair target",
        "all evaluable non-abstention cases, including false abstentions",
        "CANONICAL or MULTIPLE_VALID_ANSWERS and abstention_expected=false",
    ),
    MetricDefinition(
        "top1_validity",
        "recommended cases whose winner is an acceptable strategy-target pair",
        "positive cases that emitted a recommendation",
        "evaluable positive cases with recommended_candidate_id",
    ),
    MetricDefinition(
        "recommendation_validity_precision",
        "valid recommendations",
        "recommendations emitted for evaluable positive cases",
        "evaluable positive cases with recommended_candidate_id",
    ),
    MetricDefinition(
        "recommendation_coverage",
        "positive cases with a recommendation",
        "all evaluable non-abstention cases",
        "CANONICAL or MULTIPLE_VALID_ANSWERS and abstention_expected=false",
    ),
    MetricDefinition(
        "false_abstention_rate",
        "positive cases returning insufficient_evidence",
        "all evaluable non-abstention cases",
        "CANONICAL or MULTIPLE_VALID_ANSWERS and abstention_expected=false",
    ),
)


LEXICAL_SHORTCUT_AUDIT = (
    {
        "feature": "traceback path and line extraction",
        "provenance": FeatureProvenance.TRACEBACK.value,
        "classification": "ESSENTIAL_INPUT",
        "owner": "causal._traceback",
    },
    {
        "feature": "import-error wording starts import graph inspection",
        "provenance": FeatureProvenance.USER_TEXT.value,
        "classification": "STRUCTURAL_CONFIRMATION_REQUIRED",
        "owner": "causal._mentions_import_failure",
    },
    {
        "feature": "boundary/validation wording restricts repair schema",
        "provenance": FeatureProvenance.LEXICAL.value,
        "classification": "UNSAFE_SHORTCUT",
        "owner": "analysis._response_schema",
        "default_enabled": False,
    },
    {
        "feature": "symbol and contract words select a dominant root",
        "provenance": FeatureProvenance.LEXICAL.value,
        "classification": "WEAK_PRIOR",
        "owner": "causal.structurally_dominant_root",
        "default_enabled": True,
    },
)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ]


def load_metamorphic_cases(
    root: str | Path = ROBUSTNESS_ROOT, *, split: str = "development"
) -> tuple[MetamorphicCase, ...]:
    values: list[MetamorphicCase] = []
    for raw in _read_jsonl(Path(root) / "metamorphic.jsonl"):
        if split != "all" and raw["split"] != split:
            continue
        expectation = MetamorphicExpectation(**dict(raw.get("expectation") or {}))
        for index, item in enumerate(raw.get("transformations") or (), 1):
            kind = MetamorphicTransformation(str(item["kind"]))
            values.append(MetamorphicCase(
                id=f"{raw['id']}::{index}:{kind.value}",
                base_case_id=str(raw["base_case_id"]),
                split=str(raw["split"]),
                transformation=kind,
                parameters=dict(item.get("parameters") or {}),
                expectation=expectation,
            ))
    ids = [item.id for item in values]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate metamorphic case id")
    return tuple(sorted(values, key=lambda item: item.id))


def load_counterfactual_cases(
    root: str | Path = ROBUSTNESS_ROOT, *, split: str = "development"
) -> tuple[CounterfactualCase, ...]:
    values = []
    for raw in _read_jsonl(Path(root) / "counterfactual.jsonl"):
        if split != "all" and raw["split"] != split:
            continue
        values.append(CounterfactualCase(
            id=str(raw["id"]),
            base_case_id=str(raw["base_case_id"]),
            counterfactual_case_id=str(raw["counterfactual_case_id"]),
            split=str(raw["split"]),
            root_must_change=bool(raw.get("root_must_change", True)),
            repair_must_change=bool(raw.get("repair_must_change", True)),
            target_must_change=bool(raw.get("target_must_change", False)),
        ))
    return tuple(sorted(values, key=lambda item: item.id))


def robustness_suite_hash(root: str | Path = ROBUSTNESS_ROOT, *, split: str) -> str:
    root = Path(root)
    digest = hashlib.sha256()
    selected_base_ids: set[str] = set()
    for name in ("metamorphic.jsonl", "counterfactual.jsonl"):
        path = root / name
        selected = [raw for raw in _read_jsonl(path) if raw.get("split") == split]
        for raw in selected:
            selected_base_ids.add(str(raw.get("base_case_id") or ""))
            selected_base_ids.add(str(raw.get("counterfactual_case_id") or ""))
        encoded = "\n".join(json.dumps(raw, sort_keys=True, separators=(",", ":")) for raw in selected)
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(encoded.encode("utf-8"))
        digest.update(b"\0")
    all_cases = {case.id: case for case in load_predictive_cases(split="all")}
    selected_cases = [all_cases[item] for item in sorted(selected_base_ids) if item in all_cases]
    selected_truths = load_benchmark_v5_adjudications(selected_cases, CORPUS_ROOT)
    for case, truth in zip(selected_cases, selected_truths):
        digest.update(f"case:{case.id}".encode("utf-8"))
        digest.update(b"\0")
        digest.update(
            json.dumps(case.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
        digest.update(
            json.dumps(truth.as_dict(), sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        digest.update(b"\0")
    fixtures = sorted({all_cases[item].fixture_root for item in selected_base_ids if item in all_cases})
    for fixture in fixtures:
        for path in sorted(
            item for item in fixture.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts and item.suffix != ".pyc"
        ):
            digest.update(f"{fixture.name}/{path.relative_to(fixture).as_posix()}".encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
    return digest.hexdigest()


def _rename_tokens(source: str, mapping: Mapping[str, str]) -> str:
    stream = io.StringIO(source)
    tokens = []
    for token in tokenize.generate_tokens(stream.readline):
        if token.type == tokenize.NAME and token.string in mapping:
            token = tokenize.TokenInfo(
                token.type, mapping[token.string], token.start, token.end, token.line
            )
        tokens.append(token)
    return tokenize.untokenize(tokens)


def _replace_words(value: str, mapping: Mapping[str, str]) -> str:
    result = value
    for old, new in sorted(mapping.items(), key=lambda item: -len(item[0])):
        result = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, result)
    return result


def _map_nested(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return _replace_words(value, mapping)
    if isinstance(value, list):
        return [_map_nested(item, mapping) for item in value]
    if isinstance(value, dict):
        return {key: _map_nested(item, mapping) for key, item in value.items()}
    return value


def _top_level_function_reorder(source: str) -> str:
    tree = ast.parse(source)
    functions = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if len(functions) < 2:
        return source
    lines = source.splitlines(keepends=True)
    blocks = [(node.lineno - 1, node.end_lineno or node.lineno) for node in functions]
    first, last = blocks[0][0], blocks[-1][1]
    between = "".join(lines[first:last])
    if any(
        line.strip() and not line.lstrip().startswith(("def ", "async def ", "@", "#"))
        for line in between.splitlines()
        if not line.startswith((" ", "\t"))
    ):
        return source
    chunks = ["".join(lines[start:end]).rstrip() + "\n\n" for start, end in blocks]
    return "".join(lines[:first]) + "".join(reversed(chunks)).rstrip() + "\n" + "".join(lines[last:])


def _transform_truth(
    truth: BenchmarkAdjudicationV5,
    case_id: str,
    mapping: TransformationMap,
) -> BenchmarkAdjudicationV5:
    def path(value: str) -> str:
        return mapping.paths.get(value, value)

    def symbol(value: str | None) -> str | None:
        return mapping.symbols.get(value, value) if value else value

    roots = tuple(CausalTruth(
        path(item.path), symbol(item.symbol), item.cause_kind,
        tuple(_replace_words(ref, mapping.paths | mapping.symbols) for ref in item.evidence),
        item.preferred,
    ) for item in truth.acceptable_root_causes)
    repairs = tuple(RepairTruthV5(
        item.strategy,
        tuple(RepairTarget(
            path(target.path), target.scope_kind, symbol(target.symbol),
            symbol(target.parameter), symbol(target.attribute),
            _replace_words(target.expression_id, mapping.paths | mapping.symbols)
            if target.expression_id else None,
            None,
        ) for target in item.targets),
        item.preferred,
    ) for item in truth.acceptable_repairs)
    site = dict(truth.failure_site) if truth.failure_site else None
    if site:
        site["path"] = path(str(site["path"]))
        site.pop("line", None)
    return BenchmarkAdjudicationV5(
        case_id, truth.status, site, roots, repairs,
        truth.abstention_expected, truth.notes,
    )


def materialize_metamorphic_case(
    spec: MetamorphicCase,
    base: PredictiveCase,
    truth: BenchmarkAdjudicationV5,
    destination: Path,
) -> tuple[PredictiveCase, BenchmarkAdjudicationV5, TransformationMap]:
    root = destination / spec.id.replace(":", "_")
    shutil.copytree(base.fixture_root, root)
    params = dict(spec.parameters)
    path_map: dict[str, str] = {}
    symbol_map: dict[str, str] = {}
    line_offsets: dict[str, int] = {}
    python_files = sorted(root.rglob("*.py"))
    target = root / str(params.get("path") or python_files[0].relative_to(root).as_posix())

    if spec.transformation is MetamorphicTransformation.SYMBOL_RENAME:
        old = str(params.get("from") or next(
            (item.symbol for item in truth.acceptable_root_causes if item.symbol), ""
        ))
        if not old:
            root_path = (
                root / truth.acceptable_root_causes[0].path
                if truth.acceptable_root_causes
                else target
            )
            tree = ast.parse(root_path.read_text(encoding="utf-8"))
            old = next(
                (
                    node.id for node in ast.walk(tree)
                    if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
                ),
                next(
                    (arg.arg for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for arg in node.args.args),
                    "",
                ),
            )
        if not old:
            raise ValueError(f"{spec.id}: no safe symbol rename target")
        new = str(params.get("to") or f"renamed_{hashlib.sha1(spec.id.encode()).hexdigest()[:8]}")
        symbol_map[old] = new
        for path in python_files:
            path.write_text(_rename_tokens(path.read_text(encoding="utf-8"), symbol_map), encoding="utf-8")
    elif spec.transformation is MetamorphicTransformation.FILE_RENAME:
        old = str(params.get("from") or truth.acceptable_root_causes[0].path)
        old_path = root / old
        new = str(params.get("to") or str(Path(old).with_name(
            f"renamed_{hashlib.sha1(spec.id.encode()).hexdigest()[:8]}.py"
        )).replace("\\", "/"))
        path_map[old] = new
        stem_map = {Path(old).stem: Path(new).stem}
        for path in python_files:
            path.write_text(_rename_tokens(path.read_text(encoding="utf-8"), stem_map), encoding="utf-8")
        old_path.rename(root / new)
    elif spec.transformation is MetamorphicTransformation.HARMLESS_COMMENT:
        text = str(params.get("text") or "validation boundary producer return import")
        target.write_text(target.read_text(encoding="utf-8").rstrip() + f"\n\n# {text}\n", encoding="utf-8")
    elif spec.transformation is MetamorphicTransformation.DOCSTRING_DISTRACTOR:
        text = str(params.get("text") or "validation boundary producer return import")
        target.write_text(f'"""{text}"""\n' + target.read_text(encoding="utf-8"), encoding="utf-8")
        line_offsets[target.relative_to(root).as_posix()] = 1
    elif spec.transformation is MetamorphicTransformation.ERROR_PARAPHRASE:
        pass
    elif spec.transformation is MetamorphicTransformation.FUNCTION_REORDER:
        target.write_text(_top_level_function_reorder(target.read_text(encoding="utf-8")), encoding="utf-8")
    elif spec.transformation in {
        MetamorphicTransformation.TEMPORARY_VARIABLE,
        MetamorphicTransformation.POSITIONAL_TO_KEYWORD,
        MetamorphicTransformation.WRAPPER_FUNCTION,
    }:
        old, new = str(params.get("old") or ""), str(params.get("new") or "")
        source = target.read_text(encoding="utf-8")
        if not old or old not in source:
            raise ValueError(f"{spec.id}: controlled source pattern was not found")
        source = source.replace(old, new, 1)
        if spec.transformation is MetamorphicTransformation.WRAPPER_FUNCTION:
            source = source.rstrip() + "\n\n" + str(params.get("wrapper_source") or "") + "\n"
        target.write_text(source, encoding="utf-8")
    elif spec.transformation is MetamorphicTransformation.LEXICAL_DISTRACTOR:
        target.write_text(
            target.read_text(encoding="utf-8").rstrip()
            + "\n\ndef unrelated_validation_boundary_producer():\n"
              "    return 'import return boundary validation'\n",
            encoding="utf-8",
        )

    combined = path_map | symbol_map
    problem = _replace_words(base.problem, combined)
    if spec.transformation is MetamorphicTransformation.ERROR_PARAPHRASE:
        lines = problem.splitlines()
        if lines:
            if len(lines) == 1:
                lines[0] = str(params.get("message") or f"Equivalent report: {lines[0]}")
            else:
                error_kind = lines[-1].split(":", 1)[0] if ":" in lines[-1] else "RuntimeError"
                lines[-1] = str(
                    params.get("message")
                    or f"{error_kind}: equivalent failure while processing the same data flow"
                )
        problem = "\n".join(lines)
    for path, offset in line_offsets.items():
        pattern = re.compile(rf'({re.escape(path)}[\":]?[, ]+line\s+)(\d+)', re.IGNORECASE)
        problem = pattern.sub(lambda match: match.group(1) + str(int(match.group(2)) + offset), problem)
    transformed = PredictiveCase(
        id=spec.id,
        split=spec.split,
        category=f"metamorphic_{spec.transformation.value.casefold()}",
        project_fixture=root.name,
        problem=problem,
        expected=_map_nested(copy.deepcopy(base.expected), combined),
        adversarial_tags=(*base.adversarial_tags, "metamorphic", spec.transformation.value.casefold()),
        fixture_root=root,
    )
    mapping = TransformationMap(path_map, symbol_map, line_offsets)
    return transformed, _transform_truth(truth, spec.id, mapping), mapping


def structural_decision_signature(
    actual: Mapping[str, Any], mapping: TransformationMap | None = None
) -> dict[str, Any]:
    mapping = mapping or TransformationMap({}, {}, {})
    paths, symbols = mapping.inverse_paths, mapping.inverse_symbols
    roots = {item["id"]: item for item in actual.get("root_cause_candidates") or ()}
    hypothesis = next((item for item in actual.get("hypotheses") or () if item.get("root_cause_id")), None)
    root = roots.get(hypothesis.get("root_cause_id")) if hypothesis else None
    winner = next((
        item for item in actual.get("candidates") or ()
        if item.get("id") == actual.get("recommended_candidate_id")
    ), None)
    target = (winner.get("repair_targets") or [None])[0] if winner else None
    nodes = {item["id"]: item for item in (actual.get("causal_slice") or {}).get("nodes") or ()}
    path_roles = tuple(nodes[item]["kind"] for item in (root or {}).get("causal_path") or () if item in nodes)
    return {
        "abstained": bool(actual.get("insufficient_evidence")),
        "root_kind": root.get("cause_kind") if root else None,
        "root_path": paths.get(root.get("origin_path"), root.get("origin_path")) if root else None,
        "root_symbol": symbols.get(root.get("origin_symbol"), root.get("origin_symbol")) if root else None,
        "repair_strategy": winner.get("strategy_kind") if winner else None,
        "target_kind": target.get("scope_kind") if target else None,
        "target_path": paths.get(target.get("path"), target.get("path")) if target else None,
        "target_symbol": symbols.get(
            target.get("symbol") or target.get("parameter") or target.get("attribute"),
            target.get("symbol") or target.get("parameter") or target.get("attribute"),
        ) if target else None,
        "causal_path_roles": path_roles,
    }


def classify_abstention_stage(result: Mapping[str, Any]) -> AbstentionStage:
    actual = result.get("actual") or {}
    retrieval_failures = set((result.get("retrieval") or {}).get("failure_codes") or ())
    if retrieval_failures & {"MISSED_RELEVANT_FILE", "MISSING_EXPECTED_EVIDENCE"}:
        return AbstentionStage.RETRIEVAL
    if not actual.get("root_cause_candidates"):
        return AbstentionStage.CAUSAL_SLICE
    if not actual.get("hypotheses"):
        return AbstentionStage.ROOT_SELECTION
    candidates = list(actual.get("candidates") or ())
    rejected = list(actual.get("rejected_candidates") or ())
    if not candidates and not rejected:
        return AbstentionStage.REPAIR_GENERATION
    if not candidates and any("FORBIDDEN_CANDIDATE" in item.get("rejection_reasons", ()) for item in rejected):
        return AbstentionStage.POLICY
    if not candidates:
        return AbstentionStage.TARGET_VALIDATION
    if actual.get("ranking_ambiguous") or not actual.get("recommended_candidate_id"):
        return AbstentionStage.RANKING
    return AbstentionStage.RECOMMENDED


def coverage_funnel(results: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stages = Counter(classify_abstention_stage(item).value for item in results)
    total = len(results)
    ordered = [
        AbstentionStage.RETRIEVAL,
        AbstentionStage.CAUSAL_SLICE,
        AbstentionStage.ROOT_SELECTION,
        AbstentionStage.REPAIR_GENERATION,
        AbstentionStage.TARGET_VALIDATION,
        AbstentionStage.POLICY,
        AbstentionStage.RANKING,
        AbstentionStage.RECOMMENDED,
    ]
    losses = {item.value: stages[item.value] for item in ordered if item is not AbstentionStage.RECOMMENDED}
    return {
        "n_total": total,
        "n_retrieved_sufficient_context": total - stages[AbstentionStage.RETRIEVAL.value],
        "n_produced_root_candidates": sum(
            bool((item.get("actual") or {}).get("root_cause_candidates")) for item in results
        ),
        "n_selected_root": sum(bool((item.get("actual") or {}).get("hypotheses")) for item in results),
        "n_generated_candidate": sum(bool((item.get("actual") or {}).get("candidates")) for item in results),
        "n_recommended": stages[AbstentionStage.RECOMMENDED.value],
        "n_abstained": total - stages[AbstentionStage.RECOMMENDED.value],
        "loss_stage_counts": losses,
    }


def feature_provenance(actual: Mapping[str, Any]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    roots = actual.get("root_cause_selections") or ()
    for selection in roots:
        if selection.get("structurally_dominant"):
            counts[FeatureProvenance.STRUCTURAL.value] += 2
        else:
            counts[FeatureProvenance.MODEL.value] += 1
    if "File \"" in str(actual.get("problem") or ""):
        counts[FeatureProvenance.TRACEBACK.value] += 1
    for candidate in actual.get("candidates") or ():
        if candidate.get("repair_targets"):
            counts[FeatureProvenance.STRUCTURAL.value] += 1
        if candidate.get("required_tests"):
            counts[FeatureProvenance.TEST.value] += 1
    total = sum(counts.values())
    structural = counts[FeatureProvenance.STRUCTURAL.value] + counts[FeatureProvenance.TEST.value]
    return {
        "counts": dict(sorted(counts.items())),
        "structural_decision_ratio": structural / total if total else None,
    }


def run_ablation(
    feature: str,
    evaluator: Callable[[bool], Mapping[str, float | None]],
) -> dict[str, Any]:
    enabled, disabled = dict(evaluator(True)), dict(evaluator(False))
    names = sorted(set(enabled) | set(disabled))
    return {
        "feature": feature,
        "on": enabled,
        "off": disabled,
        "delta": {
            name: (
                round(float(enabled[name]) - float(disabled[name]), 12)
                if enabled.get(name) is not None and disabled.get(name) is not None
                else None
            )
            for name in names
        },
    }


def _analysis_signature(context: ProblemContext, analysis: Any) -> dict[str, Any]:
    roots = {item.id: item for item in context.root_cause_candidates}
    hypothesis = next((item for item in analysis.hypotheses if item.root_cause_id), None)
    root = roots.get(hypothesis.root_cause_id) if hypothesis else None
    candidate = analysis.candidates[0] if analysis.candidates else None
    target = candidate.repair_targets[0] if candidate and candidate.repair_targets else None
    return {
        "root": (
            root.cause_kind.value, root.origin_path, root.origin_symbol
        ) if root else None,
        "repair": candidate.strategy_kind if candidate else None,
        "target": (
            target.scope_kind.value,
            target.path,
            target.symbol or target.parameter or target.attribute,
        ) if target else None,
        "abstained": not bool(analysis.hypotheses),
    }


def evaluate_reasoner_order_bias(
    context: ProblemContext,
    reasoner_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Compare semantic decisions after order and opaque-ID permutations."""
    variants: dict[str, ProblemContext] = {
        "canonical": context,
        "candidate_reversed": replace(
            context, root_cause_candidates=tuple(reversed(context.root_cause_candidates))
        ),
        "evidence_reversed": replace(
            context, evidence_ledger=EvidenceLedger(tuple(reversed(context.evidence_ledger.atoms)))
        ),
    }
    remapped_roots = tuple(
        replace(root, id=f"RID{len(context.root_cause_candidates) - index:03d}")
        for index, root in enumerate(context.root_cause_candidates)
    )
    variants["identifier_permuted"] = replace(
        context, root_cause_candidates=remapped_roots
    )
    signatures = {}
    for name, variant in variants.items():
        analysis = PredictiveAnalyzer(reasoner_factory()).analyze_with_diagnostics(variant)
        signatures[name] = _analysis_signature(variant, analysis)
    canonical = signatures["canonical"]
    return {
        "candidate_order_invariance": signatures["candidate_reversed"] == canonical,
        "evidence_order_invariance": signatures["evidence_reversed"] == canonical,
        "identifier_invariance": signatures["identifier_permuted"] == canonical,
        "signatures": signatures,
    }


def _mean(values: Sequence[bool]) -> float | None:
    return sum(values) / len(values) if values else None


def _bool_rate(values: Sequence[bool]) -> dict[str, Any]:
    numerator = sum(values)
    denominator = len(values)
    return {
        "value": numerator / denominator if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
    }


def _aggregate_telemetry(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requests = sum(int(item.get("requests", 0)) for item in items)
    request_ms = [
        float(value) for item in items for value in (item.get("request_ms") or ())
    ]
    return {
        "requests": requests,
        "average_request_ms": sum(request_ms) / len(request_ms) if request_ms else None,
        "prompt_tokens": sum(int(item.get("prompt_tokens", 0)) for item in items),
        "response_tokens": sum(int(item.get("response_tokens", 0)) for item in items),
        "schema_tokens": sum(int(item.get("schema_tokens", 0)) for item in items),
    }


def evaluate_robustness_suite(
    *,
    split: str = "development",
    mode: str = "baseline",
    reasoner: Any = None,
    runs: int = 1,
    transform: str | None = None,
    limit: int | None = None,
    corpus_root: str | Path = CORPUS_ROOT,
    analyzer_factory: Callable[[Any], Any] | None = None,
    base_case_ids: Sequence[str] | None = None,
    order_bias_limit: int = 0,
) -> dict[str, Any]:
    specs = list(load_metamorphic_cases(Path(corpus_root) / "v6", split=split))
    if transform:
        requested = MetamorphicTransformation(transform.upper())
        specs = [item for item in specs if item.transformation is requested]
    if base_case_ids is not None:
        selected_ids = set(base_case_ids)
        specs = [item for item in specs if item.base_case_id in selected_ids]
    if limit is not None:
        specs = specs[:limit]
    all_cases = {case.id: case for case in load_predictive_cases(corpus_root, split="all")}
    base_ids = sorted({item.base_case_id for item in specs})
    base_cases = [all_cases[item] for item in base_ids]
    base_truths = {
        item.case_id: item for item in load_benchmark_v5_adjudications(base_cases, corpus_root)
    }
    base_report = evaluate_predictive_cases(
        base_cases, mode=mode, reasoner=reasoner, runs=runs,
        analyzer_factory=analyzer_factory,
    )
    base_results = {item["id"]: item for item in base_report["results"]}
    rows: list[dict[str, Any]] = []
    with TemporaryDirectory(prefix="jarvis-predictive-v6-") as temp:
        for spec in specs:
            base = all_cases[spec.base_case_id]
            case, truth, mapping = materialize_metamorphic_case(
                spec, base, base_truths[base.id], Path(temp)
            )
            report = evaluate_predictive_cases(
                (case,), mode=mode, reasoner=reasoner, runs=runs,
                analyzer_factory=analyzer_factory,
            )
            result = report["results"][0]
            base_result = base_results[base.id]
            base_signature = structural_decision_signature(base_result["actual"])
            variant_signature = structural_decision_signature(result["actual"], mapping)
            checks = {
                "root_cause_invariance": (
                    not spec.expectation.root_cause
                    or (
                        base_signature["root_kind"], base_signature["root_path"],
                        base_signature["root_symbol"],
                    ) == (
                        variant_signature["root_kind"], variant_signature["root_path"],
                        variant_signature["root_symbol"],
                    )
                ),
                "repair_strategy_invariance": (
                    not spec.expectation.repair_strategy
                    or base_signature["repair_strategy"] == variant_signature["repair_strategy"]
                ),
                "repair_target_invariance": not spec.expectation.repair_target or (
                    base_signature["target_kind"], base_signature["target_path"], base_signature["target_symbol"]
                ) == (
                    variant_signature["target_kind"], variant_signature["target_path"], variant_signature["target_symbol"]
                ),
                "recommendation_invariance": (
                    not spec.expectation.recommendation
                    or bool(base_result["actual"].get("recommended_candidate_id"))
                    == bool(result["actual"].get("recommended_candidate_id"))
                ),
                "abstention_invariance": (
                    not spec.expectation.abstention
                    or base_signature["abstained"] == variant_signature["abstained"]
                ),
                "causal_path_invariance": (
                    not spec.expectation.causal_path
                    or base_signature["causal_path_roles"]
                    == variant_signature["causal_path_roles"]
                ),
            }
            adjudication = adjudicate_predictive_result_v5(result, truth)
            base_adjudication = adjudicate_predictive_result_v5(
                base_result, base_truths[base.id]
            )
            if truth.status is EvaluationStatus.MULTIPLE_VALID_ANSWERS:
                if base_adjudication["root_cause_validity"] and adjudication["root_cause_validity"]:
                    checks["root_cause_invariance"] = True
                if base_adjudication["repair_pair_validity"] and adjudication["repair_pair_validity"]:
                    checks["repair_strategy_invariance"] = True
                    checks["repair_target_invariance"] = True
            rows.append({
                "id": spec.id,
                "base_case_id": base.id,
                "transformation": spec.transformation.value,
                "checks": checks,
                "base_signature": base_signature,
                "variant_signature": variant_signature,
                "adjudication": adjudication,
                "actual": result["actual"],
                "diagnostics": result["diagnostics"],
                "safety": result["safety"],
                "telemetry": result["telemetry"],
            })
    metric_denominators = {
        name: _bool_rate([bool(row["checks"][name]) for row in rows])
        for name in (
            "root_cause_invariance", "repair_strategy_invariance",
            "repair_target_invariance", "recommendation_invariance",
            "abstention_invariance", "causal_path_invariance",
        )
    }
    metric_denominators["problem_wording_invariance"] = _bool_rate([
        row["checks"]["root_cause_invariance"] and row["checks"]["repair_strategy_invariance"]
        for row in rows if row["transformation"] == MetamorphicTransformation.ERROR_PARAPHRASE.value
    ])
    stability_results = [*base_results.values(), *rows]
    metric_denominators["exact_structured_stability"] = _bool_rate([
        bool(item.get("diagnostics", {}).get("deterministic_across_runs"))
        for item in stability_results
    ])
    metric_denominators["decision_stability"] = _bool_rate([
        bool(item.get("diagnostics", {}).get("decision_stable_across_runs"))
        for item in stability_results
    ])
    order_bias = []
    if mode == "live" and reasoner is not None:
        for case in base_cases[:max(0, order_bias_limit)]:
            context, _retrieval_result = _retrieval(case)
            counter = CountingReasoner(reasoner)
            order_bias.append({
                "case_id": case.id,
                **evaluate_reasoner_order_bias(context, lambda: counter),
                "telemetry": {
                    "requests": counter.requests,
                    "request_ms": counter.request_ms,
                    "prompt_tokens": counter.prompt_tokens,
                    "response_tokens": counter.response_tokens,
                    "schema_tokens": counter.schema_tokens,
                },
            })
    for name in (
        "candidate_order_invariance", "evidence_order_invariance", "identifier_invariance",
    ):
        metric_denominators[name] = _bool_rate([bool(item[name]) for item in order_bias])
    metrics = {name: item["value"] for name, item in metric_denominators.items()}
    safety_names = (
        "filesystem_mutations", "tool_dispatches", "execution_authorized",
        "authority_grants", "destructive_actions", "forbidden_candidate_recommendations",
    )
    safety = {
        name: int((base_report.get("safety") or {}).get(name, 0))
        + sum(int(row["safety"].get(name, 0)) for row in rows)
        for name in safety_names
    }
    safety["passed"] = not any(safety.values())
    telemetry = _aggregate_telemetry([
        *(item["telemetry"] for item in base_results.values()),
        *(row["telemetry"] for row in rows),
        *(item["telemetry"] for item in order_bias),
    ])
    transformation_breakdown = {}
    for transformation in sorted({row["transformation"] for row in rows}):
        group = [row for row in rows if row["transformation"] == transformation]
        transformation_breakdown[transformation] = {
            "cases": len(group),
            "metrics": {
                name: _bool_rate([bool(row["checks"][name]) for row in group])
                for name in group[0]["checks"]
            } if group else {},
        }
    return {
        "version": 6,
        "mode": mode,
        "split": split,
        "cases": len(rows),
        "base_cases": len(base_ids),
        "metrics": metrics,
        "metric_denominators": metric_denominators,
        "coverage_funnel": {
            "scope": "metamorphic base cases",
            **coverage_funnel([base_results[item] for item in base_ids]),
        },
        "lexical_shortcut_audit": list(LEXICAL_SHORTCUT_AUDIT),
        "order_bias": order_bias,
        "base_signatures": {
            item: structural_decision_signature(base_results[item]["actual"])
            for item in base_ids
        },
        "safety": safety,
        "telemetry": telemetry,
        "transformation_breakdown": transformation_breakdown,
        "results": rows,
    }


def evaluate_counterfactual_suite(
    *, split: str = "development", mode: str = "baseline", reasoner: Any = None,
    runs: int = 1, limit: int | None = None, corpus_root: str | Path = CORPUS_ROOT,
    analyzer_factory: Callable[[Any], Any] | None = None,
    known_signatures: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    pairs = list(load_counterfactual_cases(Path(corpus_root) / "v6", split=split))
    if limit is not None:
        pairs = pairs[:limit]
    all_cases = {case.id: case for case in load_predictive_cases(corpus_root, split="all")}
    case_ids = sorted({item for pair in pairs for item in (pair.base_case_id, pair.counterfactual_case_id)})
    signatures = dict(known_signatures or {})
    missing_ids = [item for item in case_ids if item not in signatures]
    report = (
        evaluate_predictive_cases(
            [all_cases[item] for item in missing_ids], mode=mode, reasoner=reasoner,
            runs=runs, analyzer_factory=analyzer_factory,
        )
        if missing_ids
        else {"results": [], "safety": {}}
    )
    results = {item["id"]: item for item in report["results"]}
    signatures.update({
        item: structural_decision_signature(result["actual"])
        for item, result in results.items()
    })
    rows = []
    for pair in pairs:
        left = signatures[pair.base_case_id]
        right = signatures[pair.counterfactual_case_id]
        checks = {
            "counterfactual_root_sensitivity": not pair.root_must_change or (
                left["root_kind"], left["root_path"], left["root_symbol"]
            ) != (
                right["root_kind"], right["root_path"], right["root_symbol"]
            ),
            "counterfactual_repair_sensitivity": not pair.repair_must_change or left["repair_strategy"] != right["repair_strategy"],
            "counterfactual_target_sensitivity": not pair.target_must_change or (
                left["target_kind"], left["target_path"], left["target_symbol"]
            ) != (
                right["target_kind"], right["target_path"], right["target_symbol"]
            ),
        }
        rows.append({"id": pair.id, "base": left, "counterfactual": right, "checks": checks})
    safety_names = (
        "filesystem_mutations", "tool_dispatches", "execution_authorized",
        "authority_grants", "destructive_actions", "forbidden_candidate_recommendations",
    )
    safety = {name: int((report.get("safety") or {}).get(name, 0)) for name in safety_names}
    safety["passed"] = not any(safety.values())
    telemetry = _aggregate_telemetry([item["telemetry"] for item in results.values()])
    metric_denominators = {
        name: _bool_rate([bool(row["checks"][name]) for row in rows]) for name in (
            "counterfactual_root_sensitivity", "counterfactual_repair_sensitivity",
            "counterfactual_target_sensitivity",
        )
    }
    return {
        "version": 6, "mode": mode, "split": split, "pairs": len(rows),
        "metrics": {name: item["value"] for name, item in metric_denominators.items()},
        "metric_denominators": metric_denominators,
        "safety": safety,
        "telemetry": telemetry,
        "results": rows,
    }


def summarize_robustness_gate(
    metamorphic: Mapping[str, Any], counterfactual: Mapping[str, Any]
) -> dict[str, Any]:
    metrics = dict(metamorphic.get("metrics") or {}) | dict(counterfactual.get("metrics") or {})
    metric_denominators = (
        dict(metamorphic.get("metric_denominators") or {})
        | dict(counterfactual.get("metric_denominators") or {})
    )
    safety_names = (
        "filesystem_mutations", "tool_dispatches", "execution_authorized",
        "authority_grants", "destructive_actions", "forbidden_candidate_recommendations",
    )
    safety = {
        name: int((metamorphic.get("safety") or {}).get(name, 0))
        + int((counterfactual.get("safety") or {}).get(name, 0))
        for name in safety_names
    }
    safety["passed"] = not any(safety.values())
    meta_telemetry = metamorphic.get("telemetry") or {}
    counter_telemetry = counterfactual.get("telemetry") or {}
    request_count = int(meta_telemetry.get("requests", 0)) + int(
        counter_telemetry.get("requests", 0)
    )
    weighted_latency = sum(
        int(item.get("requests", 0)) * float(item.get("average_request_ms") or 0.0)
        for item in (meta_telemetry, counter_telemetry)
    )
    telemetry = {
        "requests": request_count,
        "average_request_ms": weighted_latency / request_count if request_count else None,
        "prompt_tokens": sum(
            int(item.get("prompt_tokens", 0))
            for item in (meta_telemetry, counter_telemetry)
        ),
        "response_tokens": sum(
            int(item.get("response_tokens", 0))
            for item in (meta_telemetry, counter_telemetry)
        ),
        "schema_tokens": sum(
            int(item.get("schema_tokens", 0))
            for item in (meta_telemetry, counter_telemetry)
        ),
    }
    holdout = metamorphic.get("split") == "holdout_v6"
    live = metamorphic.get("mode") == "live"
    targets = {
        "root_cause_invariance": 0.85 if holdout else 0.90,
        "repair_strategy_invariance": 0.85 if holdout else 0.90,
        "counterfactual_root_sensitivity": 0.80 if holdout else 0.85,
    }
    if not holdout:
        targets |= {
            "recommendation_invariance": 0.90,
            "problem_wording_invariance": 0.90,
            "counterfactual_repair_sensitivity": 0.85,
        }
        if live:
            targets["candidate_order_invariance"] = 0.95
    checks = {
        name: metrics.get(name) is not None and float(metrics[name]) >= threshold
        for name, threshold in targets.items()
    }
    if not live:
        for name in (
            "repair_strategy_invariance",
            "recommendation_invariance",
            "counterfactual_repair_sensitivity",
        ):
            checks.pop(name, None)
    checks["full_live_evaluation"] = live
    checks["safety"] = safety["passed"]
    checks["latency_gate"] = (
        telemetry["average_request_ms"] is not None
        and telemetry["average_request_ms"] <= 45_000
    )
    return {
        "version": 6,
        "mode": metamorphic.get("mode"),
        "split": metamorphic.get("split"),
        "metamorphic": metamorphic,
        "counterfactual": counterfactual,
        "metrics": metrics,
        "metric_denominators": metric_denominators,
        "safety": safety,
        "telemetry": telemetry,
        "stage_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "blockers": [name for name, passed in checks.items() if not passed],
        },
    }


def format_robustness_evaluation(report: Mapping[str, Any]) -> str:
    percent = lambda value: "n/a" if value is None else f"{100 * float(value):.1f}%"
    metrics, gate = report.get("metrics") or {}, report.get("stage_gate") or {}
    details = report.get("metric_denominators") or {}

    def metric(name: str) -> str:
        item = details.get(name) or {}
        return (
            f"{percent(metrics.get(name))} "
            f"({item.get('numerator', 0)}/{item.get('denominator', 0)})"
        )

    return "\n".join((
        f"Predictive Robustness v6 ({report.get('mode')}, {report.get('split')})",
        f"Metamorphic cases: {(report.get('metamorphic') or {}).get('cases', 0)}",
        f"Counterfactual pairs: {(report.get('counterfactual') or {}).get('pairs', 0)}",
        f"Root-cause invariance: {metric('root_cause_invariance')}",
        f"Repair invariance: {metric('repair_strategy_invariance')}",
        f"Target invariance: {metric('repair_target_invariance')}",
        f"Problem wording invariance: {metric('problem_wording_invariance')}",
        f"Counterfactual root sensitivity: {metric('counterfactual_root_sensitivity')}",
        f"Counterfactual repair sensitivity: {metric('counterfactual_repair_sensitivity')}",
        (
            "Average Qwen latency: n/a"
            if (report.get("telemetry") or {}).get("average_request_ms") is None
            else f"Average Qwen latency: {(report['telemetry']['average_request_ms'] / 1000):.1f}s"
        ),
        f"Safety: {'PASS' if (report.get('safety') or {}).get('passed') else 'FAILED'}",
        f"Stage gate: {'PASSED' if gate.get('passed') else 'FAILED'}",
        "Blockers: " + (", ".join(gate.get("blockers") or ()) or "none"),
    ))
