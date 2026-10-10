# External behavioral graph v14

Four independent upstream defects, two repositories, one original report per
defect. Two development and two held-out defects: this is a small architectural
study, not a representative measure of real-project diagnostic readiness.
None overlaps the observed v12/v13 corpora. No third-party code is executed.

The pinned SWE-bench Lite record and upstream corrective/test diffs adjudicate
the pre-fix symbol, effect and correction site. Diffs never enter the reasoner.
Cases with no identifiable public API keep `affected_api=null`; the evaluator
must not silently treat the corrected private symbol as an observed public API.
An operation/effect match is not proof of the expected defect-bearing relation.

For 10451, modify_field_list is one corrected location; the correction also
affects augment_descriptions_with_types. The oracle is explicitly incomplete
as a universal root/repair judge. For 8713, NumpyDocstring exposes inherited
implementation. This tests resolution limitations without treating a reported
class name as proof of a faulty helper. No expectation comes from engine output.

Acquisition: `python -m scripts.prepare_predictive_external_v14` reproduces
Python-only snapshots, verifies existing bytes and refuses to overwrite a seal.
Cases, source/patch hashes, commits and preregistered thresholds are versioned;
full traces belong in the external artifact store, not ordinary Git reports.

Holdout must run only after final runtime/evaluator freeze, exactly once.
Timeouts are censored operational observations and block readiness. Empty
conditional populations are undefined, never 100%. No tuning after holdout.
