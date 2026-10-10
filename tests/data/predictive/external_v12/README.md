# External causal evaluation v12

12 independent published defects: Requests (6), Marshmallow (2), pytest (4).
Eight development defects and four sealed holdout defects, one original issue
formulation each. Historical Requests #5667's two formulations are **one**
separate, already-observed diagnostic defect, not part of the unseen population.

Inputs and pinned pre-fix commits come from the immutable SWE-bench Lite revision
recorded in `manifest.json`. Gold corrections and added regression tests were
reviewed for adjudication; neither is passed to the reasoner. Public upstream
issue/PR URLs and patch/test hashes are recorded per defect. Missing separate
issue links use the corrective PR containing the original report; these must not
be interpreted as different defects. Erratum: Requests #1963 corrects issue
[#1955](https://github.com/psf/requests/issues/1955), not #1962 in the initial
metadata field. This reference correction does not change inputs or oracle.

Adjudication uses the changed production method's owning class (not every method
sharing its name), source locations, expected/observed behavior, and small sets
of technically acceptable strategies. Source locations are a **proxy** for root
correctness, not universal semantic proof. ExceptionInfo formatting is an API
behavior change with presentation-policy ambiguity; interpret separately rather
than asserting all formatting preferences are objective defects.

## Reproduction without executing project code

1. `python -m scripts.predictive_external_dataset` retrieves pinned public
   dataset metadata into ignored `.orchestrator/external-v12` (uses pyarrow if
   already installed, no new mandatory runtime dependency).
2. `python -m scripts.prepare_predictive_external_v12` acquires pre-fix GitHub
   archives and projects regular Python source files into ignored snapshots.
   Traversal, symlinks, large files and archive budgets are bounded. It refuses
   to overwrite a final seal. Existing seals/snapshots are checked by evaluation.
3. `python -m scripts.predictive_external_eval --split development --mode
   structural --output <new-report.json>` runs read-only structural evaluation.
4. `--mode live` explicitly opts into local Qwen. Holdout is run **once after
   hardening**, never repeatedly for tuning.

No third-party imports, installs, tests, hooks or reproduction scripts run.
Snapshots omit non-Python assets, which limits configuration/import evaluation.
Full raw decisions are verified-content-addressed gzip files in ignored
`.orchestrator/predictive-artifacts/v12`. Git retains compact per-defect
diagnostics, expected results, seals and aggregate metrics, not large graphs.

## Metrics and populations

Every rate records numerator, denominator and population. Empty populations are
undefined (`null`), never zero or 100%. Wilson 95% intervals are descriptive for
this small purposive sample, not confidence that arbitrary production projects
will behave the same. Formulations are not independent statistical observations.
Independent-defect success requires all that defect's formulations to pass.

- Retrieval: adjudicated root file appears in **initial** context.
- Correct candidate recall: source path, owning symbol and source interval match
  an available root candidate after normal recovery.
- Selection: selected roots match that oracle; conditional selection denominator
  contains only live cases where a correct candidate actually existed.
- Repair-pair validity: correct root ID, independently acceptable strategy and
  nonempty repair targets within adjudicated production paths. This is a proxy;
  target path alone does not prove a patch fixes the defect.
- Recommendation precision excludes abstentions; coverage and false abstention
  are reported alongside it, never hidden.
- Relevance precision uses adjudicated root-location correctness, **not** the
  engine's self-assessment. Selected unrelated SCCs are counted independently.
- Behavioral root validity reports only behavioral mismatch cases.
- Correct abstention is undefined here because real-defect corpus cases are
  positive. Negative/insufficient controls are covered separately by unit tests.

The external corpus is not balanced across every symptom class: there is no
independent IMPORT_INITIALIZATION defect yet. Positive import controls and
synthetic regressions prevent rejecting all import causes, but cannot establish
external import recall. Expand this population in a later independent study.

## Readiness

There is no production-readiness threshold derived from this small corpus.
Synthetic v10 approval remains distinct from external validity. Model outages,
missing roots and wrong selections must retain separate failure records.
