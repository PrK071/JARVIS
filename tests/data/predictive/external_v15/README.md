# v15 external contract population

Six new independent reports, three repositories, one original formulation per
defect. Three development and three one-shot holdout cases. The case/manifest
seal predates runtime hardening. Upstream production and test corrections are
adjudication-only and never model inputs. Some reports do not name a production
API; no gold API is inserted into ordinary runtime inputs.

Sources: [Flask #4992](https://github.com/pallets/flask/pull/4992),
[Sphinx #7975](https://github.com/sphinx-doc/sphinx/pull/7975),
[Sphinx #8506](https://github.com/sphinx-doc/sphinx/pull/8506),
[Sphinx #8627](https://github.com/sphinx-doc/sphinx/pull/8627),
[Sphinx #8801](https://github.com/sphinx-doc/sphinx/pull/8801),
[pytest #8906](https://github.com/pytest-dev/pytest/pull/8906).

Python-only pre-correction snapshots are parsed, not imported or executed.
Snapshots and full traces remain ignored; compact reports and integrity seals
are versioned. This small purposive population cannot qualify arbitrary real
projects. Hypothesis-location correctness and confirmed-root correctness must
be reported separately, including undefined empty populations and timeouts.
