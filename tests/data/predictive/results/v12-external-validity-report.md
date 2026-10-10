# JARVIS v12: external causal validity

## EXTERNAL FAILURE DECOMPOSITION

Historical Requests #5667 / PR #5673 is **one** independent encoding defect with
two formulations. The correct file was retrieved, the function was in the full
symbol index, but it was absent from the ledger and root candidate set. The
engine then chose an unrelated import SCC, fabricated an exception site and
used its own generated SCC-to-exception edges as structural dominance evidence.
There was no reported traceback or demonstrated encoding/import bridge.
This was **coverage/generation plus symptom-model failure**, not ranking between
the correct root and an incorrect root. See the preserved decomposition and
`v12-external-failure-analysis-before.md` for the per-case audit.

## EXTERNAL BENCHMARK

12 independent published defects, 12 original formulations: Requests 6,
Marshmallow 2, pytest 4. Development 8; holdout 4. Historical Requests is excluded
from both populations. Classes: exception 5, data/return contract 3, behavioral
mismatch 3, configuration behavior 1. There is no independent external import
initialization defect in this initial sample; import recall therefore cannot be
claimed externally. Positive import controls remain synthetic/unit-level.

Pre-fix source commits, original reports, corrective PRs, source archive hashes,
source projection hashes and gold correction/test hashes are recorded in the
corpus. Gold patches/test changes never enter Qwen inputs. Adjudication uses
source, upstream correction and regression-test contents, not JARVIS responses.
ExceptionInfo presentation has policy ambiguity and is explicitly documented.

Dataset revision: `6ec7bb89b9342f664a54a6e0a6ea6501d3437cc2`.
Corpus SHA-256:
`1d7f4738b2f1ffed8031f0a7e60078f37b8918663e11d2b595c02973d510f2e0`.
Freeze: `2026-10-10T16:40:01.606042+00:00`, before hardening/live evaluation.
One pre-tuning adjudication revision disambiguated methods by owning class;
its previous seal is preserved in the manifest. No holdout labels changed later.

## FROZEN ENGINE BASELINE

All rates below are **numerator / denominator**, not population accuracy claims.
Root and repair matching are independently adjudicated structural-location and
strategy proxies, **not** a mandatory LLM judge or proof of semantic entailment.

| Metric | Frozen development | After development | Holdout one-shot |
|---|---:|---:|---:|
| Relevant file retrieval | 7/8 | 7/8 | 4/4 |
| Correct root candidate recall | 1/8 | 1/8 | 0/4 |
| External root validity | 0/8 | 0/8 | 0/4 |
| Selection given correct candidate | 0/1 | 0/1 | 0/0 (undefined) |
| Repair-pair validity | 0/8 | 0/8 | 0/4 |
| Recommendation correctness | 0/6 | 0/5 | 0/1 |
| Coverage | 6/8 | 5/8 | 1/4 |
| False abstention | 2/8 | 3/8 | 3/4 |
| Irrelevant SCC selection | 4/8 | 0/8 | 0/4 |
| Behavioral root validity | 0/3 | 0/3 | 0/0 (undefined) |
| Reference validity | 49/49 | 17/17 | 6/6 |
| UNSUPPORTED grounding claims | 0/24 | 0/18 | 0/4 |
| Qwen requests | 6 | 5 | 1 |
| Average Qwen seconds/request | 40.4 | 63.1 | 73.4 |

Development selection has only **one** evaluable correct-candidate case. Holdout
selection has **none**; its zero successes must not be blamed on a selector that
never received the correct alternative. Correct abstention is undefined in this
positive external population; insufficient-evidence controls are separate tests.
Wilson 95% intervals are emitted in JSON; for 0/8 root hits the descriptive upper
bound is 32.4%, and for 0/4 it is 49.0%. This purposive, small public sample does
not estimate arbitrary-project performance. Public benchmark contamination and
Python-only projections are additional limitations.

## CAUSAL RELEVANCE

`ProblemSymptom` separates reported exception locations, behavioral reports,
initialization observations and unresolved inputs. Unknown behavioral contracts
remain unknown; no contract is invented from a function name. Behavioral anchors
use `SYMPTOM_SITE` / `AFFECTS_SYMPTOM`, not fabricated exception/traceback records.
JSON gains additive `symptom_schema_version=1` diagnostics; existing fields and
public prediction/evaluation interfaces remain available.

`CausalRelevanceAssessment` records structural truth, symptom relevance and origin
support separately. This first validator covers **SCC bridges**, not universal
Python entailment. A cycle can be detected and marked structurally true while
remaining ineligible as a causal SCC origin. Eligibility requires an observed
initialization anchor and a source-derived eager circular symbol lookup. Missing
bridges receive `INSUFFICIENT_CAUSAL_BRIDGE`, with no arbitrary relevance score.
The validator supports a conservative subset; dynamic initialization, indirect
attribute reads and complex conditional imports remain limitations. An accepted
bridge is support, not an actual execution trace or absolute proof.

## HARDENING

Repeated failure: four independent development defects selected unrelated SCCs.
Architectural cause: graph existence generated its own symptom and dominance
path. General correction: gate SCC causal edges before candidate generation and
retain typed symptom diagnostics. Measured impact: unrelated SCC selection
**4/8 -> 0/8**, a 50 percentage-point reduction. No improvement in root/repair
correctness; false abstention worsened **2/8 -> 3/8** and latency worsened. These
regressions are not hidden or renamed successful diagnoses.

One legacy unit expectation equated observing an SCC with observing an SCC-caused
defect. It was corrected; historical case labels/reports remain intact. Positive
cyclic-import tests still pass. A non-ImportError control checks that exception
wording alone is not the gate, but does not establish complete coverage of all
AttributeError initialization topologies.

No scoring weights, reasoning protocol calls, recovery budgets, evidence ledger,
grounding, policy, semantic pairwise ranking or sandbox code changed. Their
normalized byte hashes match the original freeze. No simulation/patch application
or memory was added. Non-SCC causal relevance and behavioral contract modeling
remain unresolved; generic nullable/default candidates still produce wrong roots.

## EXTERNAL HOLDOUT

Executed once after the final runtime seal and passing tests, on engine commit
`a14fdc9`. The runtime hashes exactly match `v12-runtime-freeze.json`. Results:
four missing correct candidates, three false abstentions and one wrong repair
pair. No post-holdout tuning. Relevant files were present in 4/4, but the oracle
function was represented in the ledger in only 1/4. Development ledger coverage
was 2/8. File retrieval alone substantially overstates usable causal context.

Layer distinction: **generation/symbol-context coverage** dominates. Development
has one genuine conditional selection failure (pytest cache evaluation). The
holdout provides no evaluable conditional selector population. Ranking changes
are not justified by these results.

## SYNTHETIC REGRESSION

Independent v10 structural regression: root/relation recall **172/172**, binding
identity **102/102**, SCC recall **10/10**. Stratified live regression: root,
repair-pair and top-1 validity **14/14** (12 positive cases, two expected
abstentions); recommendation precision **12/12**, import precision/recall **2/2**.
This is a regression sample, not a rerun of every historical live/holdout case.
Historical v10 reports remain unchanged. Qwen live regression mean **47.8 s**;
the operational latency backlog remains open.
The inherited v1 evaluator also emits 12 `WRONG_ROOT_CAUSE` diagnostics for these
generated cases. They use legacy location/name labels, whereas the v10 oracle
checks responsibility, structural origin and argument slots. They are preserved
separately, not counted as v10 oracle failures or silently erased. Complete raw
legacy records are in the verified regression gzip, not duplicated in Git.

## SAFETY AND TESTS

External source mutations, tool dispatches, authorization grants, execution
authorization and destructive actions: **0** in all recorded runs. Namespace
and byte checks protect new evaluations; reference audits verified preserved
full decisions independently. Forbidden recommendations: **0**. UNSUPPORTED
claims here mean reference/claim-mapping failures, **not** causal correctness.
They remain zero even when the diagnosis is wrong; relevance has its own metric.

No third-party modules, installations, reproductions or tests executed. Unknown
project code never ran on the host. Acquired archives have bounded extraction
and Windows/POSIX traversal checks. Unit tests verify a source containing an
external write instruction is parsed but never executed, and artifact output
cannot be placed inside source snapshots.

Full suite: **1524 passed, 3 skipped**. Predictive latest full-suite subset:
**283 passed, 2 skipped** (277 passed before six final acquisition/retention
boundary tests were added). A prior run had unrelated Windows lock-race and PDF
import-memory failures; both passed isolated reruns and the final full suite.
The Python 3.13 managed-environment mkdir workaround is process-local to the
test launcher, not a product/security change.

Docker is unavailable. The existing sandbox diagnostic returns all verification
capabilities false and `ready=false`; no fallback, installation, pull or privileged
configuration was performed. Follow v11's existing manual host setup and trusted
digest-image requirements, then run `jarvis predictive-sandbox-check --json`.
No Candidate Simulation was attempted.

Complete new traces: 22 verified gzip archives, **118,077,964 raw bytes** /
**17,302,363 compressed bytes**, outside regular Git artifacts. Attempts to create
a persistent store in AppData/Downloads were denied by managed permissions.
The local archives therefore remain **temporary**, not a guaranteed durable
backup. `v12-artifact-retention.json` records exact paths/hashes. A verified
portable bundle is provided for operator backup. Nothing historical was deleted
and `main` history was not rewritten. New raw traces are not uploaded by push.

## GIT

Separate benchmark and engine commits precede the holdout; stage-gate artifacts
are a third commit. Temporary projects, full traces and `_arquivo/` are excluded.
See the final delivery for the final commit and verified remote SHA.

## FINAL GATES

`PREDICTIVE_SYNTHETIC_READY = true` — only the validated synthetic consultative scope.

`PREDICTIVE_EXTERNAL_READY = false` — root generation and contract relevance are
not adequate in the measured external population; the small sample cannot justify
production readiness.

`SIMULATION_RUNTIME_READY = false` — secure runtime unavailable; simulation remains
out of scope.
