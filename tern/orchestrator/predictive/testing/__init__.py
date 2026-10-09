"""Evaluation-only helpers for generated predictive programs.

Nothing in this package is imported by the predictive runtime.  Generated
programs are benchmark inputs, not operational memory or executable patches.
"""

from .generator import (
    CausalOracle,
    GeneratedProgram,
    ProgramFamily,
    ProgramSpec,
    generate_counterfactual_pairs,
    generate_holdout_v10_supplement,
    generate_program,
    generate_suite,
    stratified_sample,
    suite_sha256,
)
from .evaluation import (
    evaluate_generated_counterfactuals,
    evaluate_generated_live,
    evaluate_generated_structural,
    evaluate_oracle_injection,
    reduce_failing_spec,
)

__all__ = [
    "CausalOracle",
    "GeneratedProgram",
    "ProgramFamily",
    "ProgramSpec",
    "generate_counterfactual_pairs",
    "generate_holdout_v10_supplement",
    "generate_program",
    "generate_suite",
    "stratified_sample",
    "suite_sha256",
    "evaluate_generated_counterfactuals",
    "evaluate_generated_live",
    "evaluate_generated_structural",
    "evaluate_oracle_injection",
    "reduce_failing_spec",
]
