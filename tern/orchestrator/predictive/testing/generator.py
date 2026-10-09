from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..evaluation import PredictiveCase


GENERATOR_VERSION = 4


class ProgramFamily(str, Enum):
    ARGUMENT_BINDING = "argument_binding"
    RETURN_CONTRACT = "return_contract"
    ARGUMENT_VS_RETURN = "argument_vs_return"
    IMPORT_GRAPH = "import_graph"
    PRODUCER_MANIFESTATION = "producer_manifestation"
    REPAIR_TARGET = "repair_target"
    NEGATIVE_CONTROL = "negative_control"


class ProgramPopulation(str, Enum):
    SINGLE_ROOT = "single_root"
    MULTIPLE_ROOT = "multiple_root"
    TRUE_ABSTENTION = "true_abstention"


@dataclass(frozen=True)
class ProgramSpec:
    family: ProgramFamily
    seed: int
    split: str = "development"
    argument_style: str = "positional"
    slot_count: int = 2
    defective_slot: int = 0
    wrapper_depth: int = 0
    temporary_variable: bool = False
    attribute_fed: bool = False
    return_fed: bool = False
    binding_valid: bool = False
    return_valid: bool = True
    contract_evidence: str = "consumer_operation"
    multiple_return_sites: bool = False
    scc_size: int = 2
    observer_import: bool = False
    type_checking_edge: bool = False
    local_import_edge: bool = False
    distractor: bool = False
    target_role: str = "CALL_SITE"

    def __post_init__(self) -> None:
        if self.seed < 0:
            raise ValueError("generation seed must be non-negative")
        if self.argument_style not in {"positional", "keyword", "mixed", "default"}:
            raise ValueError("unsupported argument style")
        if self.slot_count < 1 or not 0 <= self.defective_slot < self.slot_count:
            raise ValueError("defective slot must belong to the call signature")
        if self.wrapper_depth < 0 or self.wrapper_depth > 4:
            raise ValueError("wrapper depth must be between zero and four")
        if self.scc_size < 2 or self.scc_size > 5:
            raise ValueError("SCC size must be between two and five")

    @property
    def id(self) -> str:
        digest = hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:10]
        return f"V10-{self.split}-{self.family.value}-{digest}"

    def canonical_json(self) -> str:
        value = asdict(self)
        value["family"] = self.family.value
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @property
    def fingerprint(self) -> tuple[tuple[str, Any], ...]:
        dimensions = {
            "family": self.family.value,
            "argument_style": self.argument_style,
            "slot_count": self.slot_count,
            "defective_slot": self.defective_slot,
            "wrapper_depth": self.wrapper_depth,
            "binding_valid": self.binding_valid,
            "return_valid": self.return_valid,
            "contract_evidence": self.contract_evidence,
            "scc_size": self.scc_size,
            "observer_import": self.observer_import,
            "type_checking_edge": self.type_checking_edge,
            "local_import_edge": self.local_import_edge,
            "target_role": self.target_role,
        }
        return tuple(sorted(dimensions.items()))


@dataclass(frozen=True)
class CausalOracle:
    root_kinds: tuple[str, ...]
    responsibility_kinds: tuple[str, ...]
    origin_roles: tuple[str, ...]
    defect_bearing_relations: tuple[str, ...]
    expected_repair_strategies: tuple[str, ...]
    preferred_repair_strategy: str | None
    valid_target_roles: tuple[str, ...]
    preferred_target_role: str | None
    origin_paths: tuple[str, ...]
    origin_symbols: tuple[str, ...] = ()
    import_scc_members: tuple[str, ...] = ()
    binding_slot: int | None = None
    binding_keyword: str | None = None
    population: ProgramPopulation = ProgramPopulation.SINGLE_ROOT

    @property
    def abstention_expected(self) -> bool:
        return self.population is ProgramPopulation.TRUE_ABSTENTION

    def as_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["population"] = self.population.value
        value["abstention_expected"] = self.abstention_expected
        return value


@dataclass(frozen=True)
class GeneratedProgram:
    spec: ProgramSpec
    oracle: CausalOracle
    case: PredictiveCase
    files: Mapping[str, str] = field(repr=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.spec.id,
            "generator_version": GENERATOR_VERSION,
            "spec": json.loads(self.spec.canonical_json()),
            "oracle": self.oracle.as_dict(),
            "case": self.case.as_dict(),
            "files_sha256": hashlib.sha256(
                json.dumps(dict(self.files), sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }


_DOMAINS = (
    "media", "billing", "cache", "parser", "scheduler", "inventory",
    "renderer", "queue", "telemetry", "compression", "catalog", "delivery",
)
_VERBS = ("compose", "resolve", "emit", "hydrate", "collect", "render", "measure")
_NOUNS = ("payload", "packet", "record", "sample", "frame", "entry", "token")


def _names(spec: ProgramSpec) -> dict[str, str]:
    rng = random.Random(spec.seed)
    domain = _DOMAINS[rng.randrange(len(_DOMAINS))]
    return {
        "domain": domain,
        "module": f"{domain}_{rng.randrange(1000, 9999)}",
        "consumer": f"{_VERBS[rng.randrange(len(_VERBS))]}_{rng.randrange(10, 99)}",
        "producer": f"load_{_NOUNS[rng.randrange(len(_NOUNS))]}_{rng.randrange(10, 99)}",
        "local": f"{_NOUNS[rng.randrange(len(_NOUNS))]}_{rng.randrange(10, 99)}",
    }


def _line(source: str, needle: str) -> int:
    for number, value in enumerate(source.splitlines(), 1):
        if needle in value:
            return number
    raise AssertionError(f"generated source does not contain {needle!r}")


def _trace(path: str, source: str, caller: str, failure: str, error: str) -> str:
    caller_line = _line(source, caller)
    failure_line = _line(source, failure)
    return (
        "Traceback (most recent call last):\n"
        f'  File "{path}", line {caller_line}, in invoke\n'
        f"    {caller.strip()}\n"
        f'  File "{path}", line {failure_line}, in failure\n'
        f"    {failure.strip()}\n"
        f"{error}"
    )


def _common_expected(
    *, oracle: CausalOracle, relevant_files: Iterable[str], traceback_targets: Iterable[str]
) -> dict[str, Any]:
    files = tuple(dict.fromkeys(relevant_files))
    return {
        "insufficient_evidence": oracle.abstention_expected,
        "relevant_files": list(files),
        "acceptable_evidence": list(files),
        "relevant_tests": [],
        "traceback_targets": list(traceback_targets),
        "root_cause_signals": [],
        "acceptable_solution_families": [],
        "preferred_solution_families": [],
        "acceptable_repair_strategies": list(oracle.expected_repair_strategies),
        "preferred_repair_strategies": (
            [oracle.preferred_repair_strategy]
            if oracle.preferred_repair_strategy else []
        ),
        "repair_targets": list(oracle.origin_paths),
        "forbidden_solution_signals": ["delete file", "disable tests"],
        "v10_oracle": oracle.as_dict(),
    }


def _write_project(root: Path, files: Mapping[str, str]) -> None:
    for relative, source in files.items():
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(source, encoding="utf-8")


def _argument_program(spec: ProgramSpec, root: Path) -> GeneratedProgram:
    names = _names(spec)
    path = f"pkg/{names['module']}.py"
    params = [f"arg_{index}_{names['local']}" for index in range(spec.slot_count)]
    defaults = [
        f"{name}={'\"broken\"' if index == spec.defective_slot else index + 2}"
        if spec.argument_style == "default" and index >= spec.defective_slot
        else name
        for index, name in enumerate(params)
    ]
    expression = f"return {params[spec.defective_slot]} + 1"
    lines = [f"def {names['consumer']}({', '.join(defaults)}):", f"    {expression}", ""]
    call_target = names["consumer"]
    for depth in range(spec.wrapper_depth):
        wrapper = f"forward_{depth}_{names['domain']}"
        call_args = ", ".join(params)
        lines.extend([
            f"def {wrapper}({', '.join(params)}):",
            f"    return {call_target}({call_args})",
            "",
        ])
        call_target = wrapper
    if spec.return_fed:
        lines.extend(["def source_value():", '    return "broken"', ""])
    if spec.attribute_fed:
        lines.extend(["class InputState:", '    payload = "broken"', ""])
    values = [str(index + 2) for index in range(spec.slot_count)]
    if not spec.binding_valid:
        values[spec.defective_slot] = (
            "source_value()" if spec.return_fed else
            "InputState().payload" if spec.attribute_fed else
            '"broken"'
        )
    lines.append("def invoke():")
    if spec.temporary_variable:
        lines.append(f"    chosen = {values[spec.defective_slot]}")
        values[spec.defective_slot] = "chosen"
    if spec.argument_style == "default":
        supplied = [*values[:spec.defective_slot], *(
            f"{name}={value}"
            for name, value in zip(
                params[spec.defective_slot + 1:], values[spec.defective_slot + 1:]
            )
        )]
        call = f"return {call_target}({', '.join(supplied)})"
        responsibility = "ARGUMENT_BINDING_DEFECT"
        target_role = "PARAMETER"
        keyword = None
    elif spec.argument_style == "keyword":
        call = "return " + call_target + "(" + ", ".join(
            f"{name}={value}" for name, value in zip(params, values)
        ) + ")"
        responsibility = "ARGUMENT_SOURCE_DEFECT"
        target_role = "CALL_SITE"
        keyword = params[spec.defective_slot]
    elif spec.argument_style == "mixed":
        parts = [values[0]] + [
            f"{name}={value}" for name, value in zip(params[1:], values[1:])
        ]
        call = f"return {call_target}({', '.join(parts)})"
        responsibility = "ARGUMENT_SOURCE_DEFECT"
        target_role = "CALL_SITE"
        keyword = params[spec.defective_slot] if spec.defective_slot else None
    else:
        call = f"return {call_target}({', '.join(values)})"
        responsibility = "ARGUMENT_SOURCE_DEFECT"
        target_role = "CALL_SITE"
        keyword = None
    lines.append(f"    {call}")
    if spec.distractor:
        lines.extend([
            "",
            "def validation_return_import_boundary():",
            '    \"\"\"Unrelated producer return validation import boundary.\"\"\"',
            "    return 7",
        ])
    source = "\n".join(lines) + "\n"
    problem = _trace(
        path, source, call, expression,
        'TypeError: can only concatenate str (not "int") to str',
    )
    oracle = CausalOracle(
        root_kinds=("ARGUMENT_BINDING", "TYPE_FLOW"),
        responsibility_kinds=(responsibility,),
        origin_roles=("ARGUMENT_SOURCE",),
        defect_bearing_relations=(
            "DEFAULT_TO_FORMAL" if spec.argument_style == "default"
            else "ACTUAL_TO_FORMAL",
        ),
        expected_repair_strategies=("CORRECT_ARGUMENT",),
        preferred_repair_strategy="CORRECT_ARGUMENT",
        valid_target_roles=(target_role,),
        preferred_target_role=target_role,
        origin_paths=(path,),
        origin_symbols=(names["consumer"],),
        binding_slot=spec.defective_slot,
        binding_keyword=keyword,
    )
    expected = _common_expected(
        oracle=oracle, relevant_files=(path,),
        traceback_targets=(f"{path}:{_line(source, expression)}",),
    )
    case = PredictiveCase(
        id=spec.id, split=spec.split, category=spec.family.value,
        project_fixture=spec.id, problem=problem, expected=expected,
        adversarial_tags=("distractor",) if spec.distractor else (), fixture_root=root,
    )
    files = {path: source, "pkg/__init__.py": ""}
    _write_project(root, files)
    return GeneratedProgram(spec, oracle, case, files)


def _return_program(spec: ProgramSpec, root: Path) -> GeneratedProgram:
    names = _names(spec)
    producer_path = f"pkg/{names['module']}_source.py"
    consumer_path = f"pkg/{names['module']}_use.py"
    annotation = " -> int" if spec.contract_evidence == "type_annotation" else ""
    bad = "7" if spec.return_valid else '"broken"'
    producer_lines = [f"def {names['producer']}(){annotation}:"]
    if spec.multiple_return_sites:
        producer_lines.extend(["    if False:", "        return 7"])
    producer_lines.append(f"    return {bad}")
    producer = "\n".join(producer_lines) + "\n"
    target = names["producer"]
    consumer_lines = [f"from .{names['module']}_source import {names['producer']}", ""]
    for depth in range(spec.wrapper_depth):
        wrapper = f"relay_{depth}_{names['domain']}"
        consumer_lines.extend([f"def {wrapper}():", f"    return {target}()", ""])
        target = wrapper
    consumer_lines.extend([
        "def failure(value):",
        "    return value + 1",
        "",
        "def invoke():",
        f"    value = {target}()",
        "    return failure(value)",
    ])
    if spec.distractor:
        consumer_lines.extend([
            "", "def argument_boundary_import():", '    return "return validation"',
        ])
    consumer = "\n".join(consumer_lines) + "\n"
    call = "return failure(value)"
    failure = "return value + 1"
    problem = _trace(
        consumer_path, consumer, call, failure,
        'TypeError: can only concatenate str (not "int") to str',
    )
    population = (
        ProgramPopulation.TRUE_ABSTENTION if spec.return_valid
        else ProgramPopulation.SINGLE_ROOT
    )
    oracle = CausalOracle(
        root_kinds=() if spec.return_valid else ("RETURN_CONTRACT", "TYPE_FLOW"),
        responsibility_kinds=() if spec.return_valid else ("RETURN_CONTRACT_DEFECT",),
        origin_roles=() if spec.return_valid else ("RETURN_SOURCE", "PRODUCER"),
        defect_bearing_relations=() if spec.return_valid else ("RETURN_TO_CONSUMER",),
        expected_repair_strategies=() if spec.return_valid else ("CORRECT_RETURN_VALUE", "FIX_PRODUCER"),
        preferred_repair_strategy=None if spec.return_valid else "CORRECT_RETURN_VALUE",
        valid_target_roles=() if spec.return_valid else ("RETURN_SITE", "FUNCTION"),
        preferred_target_role=None if spec.return_valid else "RETURN_SITE",
        origin_paths=() if spec.return_valid else (producer_path,),
        origin_symbols=() if spec.return_valid else (names["producer"],),
        population=population,
    )
    expected = _common_expected(
        oracle=oracle,
        relevant_files=(consumer_path,) if spec.return_valid else (consumer_path, producer_path),
        traceback_targets=(f"{consumer_path}:{_line(consumer, failure)}",),
    )
    case = PredictiveCase(
        id=spec.id, split=spec.split, category=spec.family.value,
        project_fixture=spec.id, problem=problem, expected=expected,
        adversarial_tags=("distractor",) if spec.distractor else (), fixture_root=root,
    )
    files = {
        producer_path: producer, consumer_path: consumer, "pkg/__init__.py": "",
    }
    _write_project(root, files)
    return GeneratedProgram(spec, oracle, case, files)


def _factorial_program(spec: ProgramSpec, root: Path) -> GeneratedProgram:
    names = _names(spec)
    path = f"pkg/{names['module']}.py"
    produced = "8" if spec.return_valid else '"broken_return"'
    bound = "2" if spec.binding_valid else '"broken_argument"'
    lines = [
        f"def {names['producer']}() -> int:",
        f"    return {produced}",
        "",
        f"def {names['consumer']}(produced, supplied):",
        "    return produced / supplied",
        "",
    ]
    producer_target = names["producer"]
    for depth in range(spec.wrapper_depth):
        wrapper = f"relay_{depth}_{names['domain']}"
        lines.extend([f"def {wrapper}():", f"    return {producer_target}()", ""])
        producer_target = wrapper
    call = f"return {names['consumer']}({producer_target}(), {bound})"
    lines.extend(["def invoke():", f"    {call}"])
    source = "\n".join(lines) + "\n"
    failure = "return produced / supplied"
    problem = (
        _trace(path, source, call, failure, "TypeError: unsupported operand type(s) for /")
        if not (spec.binding_valid and spec.return_valid)
        else "Reported arithmetic issue without a traceback or reproducible failing value."
    )
    roots: list[str] = []
    responsibilities: list[str] = []
    strategies: list[str] = []
    targets: list[str] = []
    if not spec.return_valid:
        roots.extend(("RETURN_CONTRACT", "TYPE_FLOW"))
        responsibilities.append("RETURN_CONTRACT_DEFECT")
        strategies.extend(("CORRECT_RETURN_VALUE", "FIX_PRODUCER"))
        targets.extend(("RETURN_SITE", "FUNCTION"))
    if not spec.binding_valid:
        roots.extend(("ARGUMENT_BINDING", "TYPE_FLOW"))
        responsibilities.append("ARGUMENT_SOURCE_DEFECT")
        strategies.append("CORRECT_ARGUMENT")
        targets.append("CALL_SITE")
    invalid_count = int(not spec.return_valid) + int(not spec.binding_valid)
    population = (
        ProgramPopulation.TRUE_ABSTENTION if invalid_count == 0 else
        ProgramPopulation.MULTIPLE_ROOT if invalid_count == 2 else
        ProgramPopulation.SINGLE_ROOT
    )
    oracle = CausalOracle(
        root_kinds=tuple(dict.fromkeys(roots)),
        responsibility_kinds=tuple(responsibilities),
        origin_roles=("RETURN_SOURCE", "ARGUMENT_SOURCE") if invalid_count == 2 else
            (("RETURN_SOURCE",) if not spec.return_valid else ("ARGUMENT_SOURCE",)),
        defect_bearing_relations=tuple(
            relation for condition, relation in (
                (not spec.return_valid, "RETURN_TO_CONSUMER"),
                (not spec.binding_valid, "ACTUAL_TO_FORMAL"),
            ) if condition
        ),
        expected_repair_strategies=tuple(dict.fromkeys(strategies)),
        preferred_repair_strategy=(
            None if invalid_count != 1 else
            "CORRECT_RETURN_VALUE" if not spec.return_valid else "CORRECT_ARGUMENT"
        ),
        valid_target_roles=tuple(dict.fromkeys(targets)),
        preferred_target_role=(
            None if invalid_count != 1 else
            "RETURN_SITE" if not spec.return_valid else "CALL_SITE"
        ),
        origin_paths=() if invalid_count == 0 else (path,),
        origin_symbols=tuple(
            value for condition, value in (
                (not spec.return_valid, names["producer"]),
                (not spec.binding_valid, names["consumer"]),
            ) if condition
        ),
        binding_slot=1 if not spec.binding_valid else None,
        population=population,
    )
    expected = _common_expected(
        oracle=oracle, relevant_files=(path,),
        traceback_targets=(f"{path}:{_line(source, failure)}",),
    )
    case = PredictiveCase(
        id=spec.id, split=spec.split, category=spec.family.value,
        project_fixture=spec.id, problem=problem, expected=expected,
        adversarial_tags=(), fixture_root=root,
    )
    files = {path: source, "pkg/__init__.py": ""}
    _write_project(root, files)
    return GeneratedProgram(spec, oracle, case, files)


def _import_program(spec: ProgramSpec, root: Path) -> GeneratedProgram:
    names = _names(spec)
    modules = [f"pkg/{names['domain']}_{index}.py" for index in range(spec.scc_size)]
    cycle = spec.family is not ProgramFamily.NEGATIVE_CONTROL
    files: dict[str, str] = {"pkg/__init__.py": ""}
    for index, path in enumerate(modules):
        next_index = index + 1
        lines: list[str] = []
        if next_index < len(modules):
            lines.append(f"from .{Path(modules[next_index]).stem} import item_{next_index}")
        elif cycle:
            if spec.local_import_edge:
                lines.extend([
                    "def resolve_last():",
                    f"    from .{Path(modules[0]).stem} import item_0",
                    "    return item_0",
                ])
            elif spec.type_checking_edge:
                lines.extend([
                    "from typing import TYPE_CHECKING",
                    "if TYPE_CHECKING:",
                    f"    from .{Path(modules[0]).stem} import item_0",
                ])
            else:
                lines.append(f"from .{Path(modules[0]).stem} import item_0")
        lines.extend(["", f"item_{index} = {index}"])
        files[path] = "\n".join(lines) + "\n"
    if spec.observer_import:
        files["tests/test_observer.py"] = (
            f"from pkg.{Path(modules[0]).stem} import item_0\n\n"
            "def test_observer():\n    assert item_0 == 0\n"
        )
    runtime_cycle = cycle and not spec.type_checking_edge and not spec.local_import_edge
    problem = (
        "Traceback (most recent call last):\n"
        f'  File "{modules[0]}", line 1, in <module>\n'
        f"ImportError: cannot import name 'item_0' from partially initialized module "
        f"'pkg.{Path(modules[0]).stem}'"
    )
    oracle = CausalOracle(
        root_kinds=("IMPORT_RESOLUTION",) if runtime_cycle else (),
        responsibility_kinds=("IMPORT_GRAPH_DEFECT",) if runtime_cycle else (),
        origin_roles=("IMPORT_SCC",) if runtime_cycle else (),
        defect_bearing_relations=("PRODUCTION_IMPORT_CYCLE",) if runtime_cycle else (),
        expected_repair_strategies=("CORRECT_IMPORT",) if runtime_cycle else (),
        preferred_repair_strategy="CORRECT_IMPORT" if runtime_cycle else None,
        valid_target_roles=("IMPORT_EDGE", "MODULE") if runtime_cycle else (),
        preferred_target_role="IMPORT_EDGE" if runtime_cycle else None,
        origin_paths=tuple(modules) if runtime_cycle else (),
        import_scc_members=tuple(modules) if runtime_cycle else (),
        population=(ProgramPopulation.SINGLE_ROOT if runtime_cycle else ProgramPopulation.TRUE_ABSTENTION),
    )
    expected = _common_expected(
        oracle=oracle, relevant_files=(modules[0],),
        traceback_targets=(f"{modules[0]}:1",),
    )
    case = PredictiveCase(
        id=spec.id, split=spec.split, category=spec.family.value,
        project_fixture=spec.id, problem=problem, expected=expected,
        adversarial_tags=("observer_import",) if spec.observer_import else (), fixture_root=root,
    )
    _write_project(root, files)
    return GeneratedProgram(spec, oracle, case, files)


def _producer_program(spec: ProgramSpec, root: Path) -> GeneratedProgram:
    return _return_program(replace(spec, family=ProgramFamily.PRODUCER_MANIFESTATION), root)


def _target_program(spec: ProgramSpec, root: Path) -> GeneratedProgram:
    if spec.target_role in {"RETURN_SITE", "FUNCTION"}:
        return _return_program(replace(spec, family=ProgramFamily.REPAIR_TARGET, return_valid=False), root)
    if spec.target_role == "IMPORT_EDGE":
        return _import_program(replace(spec, family=ProgramFamily.REPAIR_TARGET), root)
    return _argument_program(replace(spec, family=ProgramFamily.REPAIR_TARGET), root)


def generate_program(spec: ProgramSpec, root: Path) -> GeneratedProgram:
    root.mkdir(parents=True, exist_ok=True)
    if spec.family is ProgramFamily.ARGUMENT_BINDING:
        return _argument_program(spec, root)
    if spec.family is ProgramFamily.RETURN_CONTRACT:
        return _return_program(spec, root)
    if spec.family is ProgramFamily.ARGUMENT_VS_RETURN:
        return _factorial_program(spec, root)
    if spec.family is ProgramFamily.IMPORT_GRAPH:
        return _import_program(spec, root)
    if spec.family is ProgramFamily.PRODUCER_MANIFESTATION:
        return _producer_program(spec, root)
    if spec.family is ProgramFamily.REPAIR_TARGET:
        return _target_program(spec, root)
    if spec.family is ProgramFamily.NEGATIVE_CONTROL:
        if spec.scc_size > 1 and (spec.observer_import or spec.type_checking_edge):
            return _import_program(spec, root)
        return _return_program(replace(spec, return_valid=True), root)
    raise AssertionError(f"unsupported generated family: {spec.family}")


def generate_holdout_v10_supplement() -> tuple[ProgramSpec, ...]:
    """Return the sealed tail omitted by the original 48-case positional cap.

    The first one-shot exposed zero denominators for import SCC and
    manifestation controls.  These are the already-defined, previously
    unexecuted seeds immediately following that cap; no observed model answer
    is used to select them.
    """
    split = "holdout_v10_supplement"
    specs: list[ProgramSpec] = [
        ProgramSpec(
            ProgramFamily.ARGUMENT_VS_RETURN, seed, split,
            binding_valid=True, return_valid=True, wrapper_depth=depth,
        )
        for seed, depth in ((3048, 1), (3049, 3))
    ]
    seed = 3050
    for size in (2, 4):
        for observer in (False, True):
            specs.append(ProgramSpec(
                ProgramFamily.IMPORT_GRAPH, seed, split, scc_size=size,
                observer_import=observer,
            ))
            seed += 1
        for edge_kind in ("type", "local"):
            specs.append(ProgramSpec(
                ProgramFamily.NEGATIVE_CONTROL, seed, split, scc_size=size,
                observer_import=True, type_checking_edge=edge_kind == "type",
                local_import_edge=edge_kind == "local",
            ))
            seed += 1
    for depth in (1, 3):
        specs.append(ProgramSpec(
            ProgramFamily.PRODUCER_MANIFESTATION, seed, split,
            wrapper_depth=depth, return_valid=False,
        ))
        seed += 1
    return tuple(specs)


def generate_suite(split: str = "development") -> tuple[ProgramSpec, ...]:
    if split == "holdout_v10_supplement":
        return generate_holdout_v10_supplement()
    specs: list[ProgramSpec] = []
    base_seed = {"development": 1000, "compositional": 2000, "holdout_v10": 3000}[split]
    if split == "development":
        wrappers, scc_sizes = (0, 1, 2), (2, 3)
        styles = ("positional", "keyword", "default")
    elif split == "compositional":
        wrappers, scc_sizes = (2, 3), (3, 4)
        styles = ("mixed", "default")
    else:
        wrappers, scc_sizes = (1, 3), (2, 4)
        styles = ("mixed", "keyword", "default")
    seed = base_seed
    for style in styles:
        for depth in wrappers:
            for slot_count in (2, 3):
                for defective in range(slot_count):
                    specs.append(ProgramSpec(
                        ProgramFamily.ARGUMENT_BINDING, seed, split,
                        argument_style=style, slot_count=slot_count,
                        defective_slot=defective, wrapper_depth=depth,
                        temporary_variable=bool(seed % 2), distractor=bool(seed % 3 == 0),
                    ))
                    seed += 1
    for depth in wrappers:
        for evidence in ("type_annotation", "consumer_operation", "test_assertion"):
            for multiple in (False, True):
                specs.append(ProgramSpec(
                    ProgramFamily.RETURN_CONTRACT, seed, split,
                    wrapper_depth=depth, return_valid=False,
                    contract_evidence=evidence, multiple_return_sites=multiple,
                    distractor=bool(seed % 2),
                ))
                seed += 1
    for binding_valid in (False, True):
        for return_valid in (False, True):
            for depth in wrappers[:2]:
                specs.append(ProgramSpec(
                    ProgramFamily.ARGUMENT_VS_RETURN, seed, split,
                    binding_valid=binding_valid, return_valid=return_valid,
                    wrapper_depth=depth,
                ))
                seed += 1
    for size in scc_sizes:
        for observer in (False, True):
            specs.append(ProgramSpec(
                ProgramFamily.IMPORT_GRAPH, seed, split, scc_size=size,
                observer_import=observer,
            ))
            seed += 1
        for edge_kind in ("type", "local"):
            specs.append(ProgramSpec(
                ProgramFamily.NEGATIVE_CONTROL, seed, split, scc_size=size,
                observer_import=True, type_checking_edge=edge_kind == "type",
                local_import_edge=edge_kind == "local",
            ))
            seed += 1
    for depth in wrappers:
        specs.append(ProgramSpec(
            ProgramFamily.PRODUCER_MANIFESTATION, seed, split,
            wrapper_depth=depth, return_valid=False,
        ))
        seed += 1
    for role in ("CALL_SITE", "PARAMETER", "RETURN_SITE", "IMPORT_EDGE"):
        specs.append(ProgramSpec(
            ProgramFamily.REPAIR_TARGET, seed, split, target_role=role,
            argument_style="default" if role == "PARAMETER" else "positional",
        ))
        seed += 1
    if split == "development":
        # A second naming/surface draw for every structural specification proves
        # that offline accuracy is not an accident of one vocabulary.  The
        # causal factors stay identical while names, comments and local layout
        # change through the seed.
        originals = tuple(specs)
        specs.extend(
            replace(item, seed=item.seed + 500, distractor=not item.distractor)
            for item in originals
        )
    # Keep the sealed set affordable while preserving unseen combinations.
    limit = 220 if split == "development" else 60 if split == "compositional" else 48
    return tuple(specs[:limit])


def generate_counterfactual_pairs(
    *, count: int = 24, split: str = "development"
) -> tuple[tuple[ProgramSpec, ProgramSpec], ...]:
    pairs: list[tuple[ProgramSpec, ProgramSpec]] = []
    for index in range(count):
        base = ProgramSpec(
            ProgramFamily.ARGUMENT_VS_RETURN, 7000 + index * 2, split,
            binding_valid=False, return_valid=True,
            argument_style=("positional", "keyword", "mixed")[index % 3],
            wrapper_depth=index % 3,
        )
        counterfactual = replace(
            base, seed=base.seed + 1, binding_valid=True, return_valid=False,
        )
        pairs.append((base, counterfactual))
    return tuple(pairs)


def stratified_sample(
    specs: Iterable[ProgramSpec], limit: int
) -> tuple[ProgramSpec, ...]:
    """Round-robin families and then structural fingerprints deterministically."""
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    buckets: dict[ProgramFamily, list[ProgramSpec]] = {}
    for spec in specs:
        buckets.setdefault(spec.family, []).append(spec)
    for family, values in buckets.items():
        remaining = list(values)
        ordered: list[ProgramSpec] = []
        seen: dict[str, set[Any]] = {}

        def features(item: ProgramSpec) -> tuple[tuple[str, Any], ...]:
            base = list(item.fingerprint)
            if family is ProgramFamily.ARGUMENT_VS_RETURN:
                base.append(("validity_quadrant", (item.binding_valid, item.return_valid)))
            elif family is ProgramFamily.ARGUMENT_BINDING:
                base.append((
                    "binding_topology",
                    (item.argument_style, item.wrapper_depth, item.defective_slot),
                ))
            elif family is ProgramFamily.IMPORT_GRAPH:
                base.append((
                    "import_topology",
                    (item.scc_size, item.observer_import, item.type_checking_edge, item.local_import_edge),
                ))
            return tuple(base)

        while remaining:
            def novelty(item: ProgramSpec) -> tuple[int, str]:
                return (
                    sum(
                        value not in seen.get(name, set())
                        for name, value in features(item)
                    ),
                    item.id,
                )

            chosen = max(remaining, key=novelty)
            remaining.remove(chosen)
            ordered.append(chosen)
            for name, value in features(chosen):
                seen.setdefault(name, set()).add(value)
        buckets[family] = ordered
    selected: list[ProgramSpec] = []
    families = sorted(buckets, key=lambda item: item.value)
    while len(selected) < limit and any(buckets.values()):
        for family in families:
            if buckets[family] and len(selected) < limit:
                selected.append(buckets[family].pop(0))
    return tuple(selected)


def suite_sha256(specs: Iterable[ProgramSpec]) -> str:
    payload = {
        "generator_version": GENERATOR_VERSION,
        "specs": [json.loads(item.canonical_json()) for item in specs],
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
