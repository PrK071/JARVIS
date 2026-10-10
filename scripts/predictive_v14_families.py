"""Avaliacao estrutural de familias; nao importa/executa fixtures gerados."""
import argparse
import hashlib
import json
import subprocess
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from tern.orchestrator.predictive.causal import build_causal_slice
from tern.orchestrator.predictive.external_evaluation import rate
from tern.orchestrator.predictive.grounding import build_evidence_ledger
from tern.orchestrator.predictive.service import build_problem_context
from tern.orchestrator.predictive.testing.behavioral_families import behavioral_families
from tern.orchestrator.project_intelligence_v2 import ProjectCandidateGenerator, ProjectIndexBuilderV2
from tern.orchestrator.security import PathPolicy


def evaluate(workspace, *, graph_enabled=False):
    rows = []
    for family in behavioral_families():
        root = workspace / family.name
        for path, source in family.files.items():
            dest = root / path
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and dest.read_text() != source:
                raise ValueError("Fixture existente divergente.")
            dest.write_text(source)
        policy = PathPolicy((root.resolve(),))
        snapshot = ProjectIndexBuilderV2(root, path_policy=policy).build()
        context = build_problem_context(family.problem, snapshot,
            ProjectCandidateGenerator().generate(family.problem, snapshot), policy)
        context = replace(context, evidence_ledger=build_evidence_ledger(context, snapshot, policy))
        if graph_enabled:
            from tern.orchestrator.predictive.behavioral import bind_graph_evidence
            context = replace(context, behavioral_graph=bind_graph_evidence(
                context.behavioral_graph, context.evidence_ledger))
        causal, roots = build_causal_slice(context, snapshot, policy)
        selected = {item.qualified_name for item in context.relevant_symbol_contexts}
        # v13 only exposes selected symbols, not a demonstrated implementation
        # graph. This transport proxy is never labeled causal representability.
        row = {"family": family.name, "api_selected": family.api in selected,
            "implementation_symbol_proxy": family.implementation in selected,
            "generated_root_count": len(roots), "contract_supported": None,
            "defect_relation_representable": None}
        if graph_enabled:
            graph = context.behavioral_graph
            row.update(implementation_path_available=bool(graph and family.implementation
                and any(item.qualified_name == family.implementation for item in graph.callables)),
                behavioral_effect_available=bool(graph and family.effect
                and any(item.kind.value == family.effect for item in graph.effects)),
                graph=graph.as_dict() if graph else None)
        rows.append(row)
    eligible_path = [row for row, family in zip(rows, behavioral_families()) if family.implementation]
    eligible_effect = [row for row, family in zip(rows, behavioral_families()) if family.effect]
    return {"version": 14, "mode": "behavioral_graph" if graph_enabled else "v13_frozen",
        "cases": rows, "independent_families": len(rows),
        "metrics": {"api_selected": rate(sum(r['api_selected'] for r in rows), len(rows), "family controls"),
            "implementation_symbol_proxy": rate(sum(r['implementation_symbol_proxy'] for r in eligible_path), len(eligible_path), "known implementation controls"),
            **({"implementation_path_available": rate(sum(r['implementation_path_available'] for r in eligible_path), len(eligible_path), "known implementation controls"),
                "behavioral_effect_available": rate(sum(r['behavioral_effect_available'] for r in eligible_effect), len(eligible_effect), "known effect controls")}
                if graph_enabled else {})},
        "limitation": "Reachability/effects are not defect responsibility. Controls do not encode a violated contract."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--graph', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Resultado existente.")
    result = evaluate(Path('.orchestrator/predictive-v14-families'), graph_enabled=args.graph)
    result['engine_commit'] = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    result['frozen_at'] = datetime.now(timezone.utc).isoformat()
    result['runtime_hashes'] = {str(p): hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest()
        for p in sorted(Path('tern/orchestrator/predictive').glob('*.py'))}
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result['metrics']))


if __name__ == '__main__':
    main()
