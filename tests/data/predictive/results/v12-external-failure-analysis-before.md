# External failure decomposition before hardening

Engine: `601b4e697a0d0ee6d79ca716939e37a41d293902`.
Corpus seal: `1d7f4738b2f1ffed8031f0a7e60078f37b8918663e11d2b595c02973d510f2e0`.
All observations below precede engine changes. Holdout has not been evaluated.

## Historical Requests #5667 (one defect, two formulations)

The upstream correction adds UTF-8 fallback for JSON without a declared charset
in `requests/utils.py::get_encoding_from_headers`
([issue](https://github.com/psf/requests/issues/5667),
[correction](https://github.com/psf/requests/pull/5673)).
Both preserved traces show:

1. Initial retrieval includes `requests/utils.py` among 12 selected files.
2. The correct function exists in Project Intelligence's full symbol index.
3. No ledger atom covers that function; no root candidate names it.
4. There is no reported traceback.
5. `build_causal_slice` chooses the first production edge of a retrieved SCC as
   a failure seed. It creates `EXCEPTION_SITE` at `requests/__init__.py:123`.
6. It creates deterministic `RAISES_AT` edges from SCC/imports to that synthetic
   exception. They do not derive from a reported exception or encoding flow.
7. The only SCC root has a nominal complete path, and structural dominance
   constrains the reasoner's selectable IDs to it.
8. Grounding canonicalizes real graph facts, but those facts do not establish
   relevance to decoding. Zero unsupported *reference-backed* claims does not
   mean the causal conclusion was supported.

Mechanism: **symbol coverage / generation failure plus fabricated causal bridge**,
not ranking between a correct encoding root and an incorrect SCC root. The
correct root was absent. Full bridge references are in
`v12-requests-decomposition.json`; complete original traces remain preserved by
the v11 artifact-retention policy.

## Independent development population

| Defect | Correct file initially? | Correct root available? | Observed failure |
|---|---|---|---|
| Marshmallow 1359 | yes | no | missing root / abstention |
| Requests 1963 | yes | no | irrelevant SCC dominates |
| Requests 2317 | yes | no | irrelevant SCC dominates |
| Requests 2674 | yes | no | irrelevant SCC dominates |
| Requests 3362 | no | no | retrieval miss + irrelevant SCC |
| pytest 11143 | yes | no | missing root / abstention |
| pytest 5413 | yes | no | unrelated roots / wrong repair |
| pytest 7373 | yes | yes | correct root exists but selection loses |

Rates: retrieval **7/8**, correct candidate recall **1/8**, root validity **0/8**,
repair-pair validity **0/8**, recommendation correctness **0/6**, coverage **6/8**,
false abstention **2/8**, unrelated SCC selection **4/8**. All 8 defects have one
formulation. Conditional root selection is **0/1**, not 0/8. These populations
must not be compared silently to the v10 synthetic oracle.

## Generalizable correction justified

Four independent development defects reproduce the unrelated-SCC mechanism.
The proposed correction is limited to requiring an observed initialization
anchor plus an eager circular symbol lookup before linking SCC to a symptom.
Real cycles remain detectable. Positive controls include initialization
AttributeError, not only ImportError. Negative controls include SCC plus logic
defect, consumer traceback, benign module cycle and misleading import wording.

No general scoring/recovery changes are justified by these observations. A
relevance gate can reduce confident misattribution, but cannot itself recover
missing behavioral roots. Increased abstention must be reported as a limitation,
not renamed successful diagnosis.
