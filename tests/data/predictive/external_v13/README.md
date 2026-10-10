# External symbol coverage v13

Ten new independent defects: six development, four sealed holdout, three
repositories (pytest, Sphinx, Flask). None duplicates the twelve observed v12
defects. Each original report is one formulation, not a synthetic prompt with
the answer inserted. Source acquisition is Python-only and read-only; no
third-party module, installer or test is executed on the host.

Reproduce acquisition from the pinned v12 dataset cache with
`python -m scripts.prepare_predictive_external_v13`. The script refuses to
overwrite a divergent seal or snapshot. Corrective patches are used solely for
adjudication; only hashes and source locations are retained in the corpus.

Use `scripts.predictive_external_eval` with `--corpus`, `--snapshots`, `--archive`,
`--split` and `--mode structural|live`. Live is explicit opt-in. Holdout is not
evaluated during hardening. Reports preserve the evaluator version, source
hashes and content-addressed compressed traces outside ordinary Git artifacts.

## Metric contract

Every rate carries numerator, denominator, population and descriptive Wilson
interval. File retrieval is not symbol retrieval. A broad class definition is
not proof that a method's statements entered the ledger. Local AST relation
coverage is a representation proxy, not semantic proof of the defect. Root
candidate recall requires adjudicated path, owning symbol and source range;
selection accuracy is conditional on such a candidate existing.

Conversion rates use only observations passing the source stage. Unobserved
model stages in structural mode are `null`, never interpreted as false or
perfect. Abstentions stay in positive end-to-end populations. This purposive
small sample cannot establish production readiness even if thresholds pass.
