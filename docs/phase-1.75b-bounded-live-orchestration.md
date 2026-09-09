# PHASE 1.75B — BOUNDED LIVE ORCHESTRATION

STATUS: **APPROVED — BOUNDED AUTONOMY VERIFIED**

## ARCHITECTURE

```text
UserGoal + factual WorldState
        ↓
OrchestrationPolicy (Qwen chooses one next action)
        ↓
NextActionValidator
        ↓
BoundedLiveExecutionAuthority (permission only)
        ↓ ALLOW                         ↓ BLOCK
LiveActionSink → ToolRegistry          Observation
        ↓                                  ↓
Observation ─→ deterministic WorldStateReducer
        ↓
replan / verify / respond
```

`ORCHESTRATION_MODE=shadow` remains the rollback/evaluation mode.
`ORCHESTRATION_MODE=bounded_live` is the only mode that connects the loop to
real tools and agents. Model output has no direct executor handle.

LIVE ORCHESTRATION: **enabled behind explicit mode configuration**

REAL ACTION TYPES ENABLED:

- `INSPECT`: project resolution, file/list/search, Git state, sessions, jobs,
  web reads and other allow-listed read-only tools.
- `EXECUTE`: safe tests/static checks and authority-approved local writes.
- `DELEGATE`: real Codex READ_ONLY/MUTATION and DeepSeek READ_ONLY dispatch.
- `WAIT`, `ASK_USER`, `RESPOND`, `STOP`: bounded control actions.

## RISK MATRIX

| Effect | Disposition |
|---|---|
| READ, SEARCH, INSPECT, QUERY_STATE, CHECK_GIT, CHECK_DIFF, RUN_SAFE_TESTS | AUTO |
| DeepSeek analysis, Codex READ_ONLY | AUTO after capability/availability gates |
| Codex MUTATION, local create/write in allowed scope | AUTHORITY |
| delete, application scheduling, sensitive external effects | CONFIRM |
| unknown/unlisted effects | RESTRICTED |

Direct overwrite still requires confirmation outside bounded-live. In
bounded-live, a concrete in-process authority decision with
`mutation_authorized=true` removes only the redundant medium-risk overwrite
prompt. Delete/high-risk confirmation is unchanged.

## LIVE EVALUATION

Safe real runs (n=4): Git inspection, isolated repair, Codex READ_ONLY and
DeepSeek READ_ONLY. The repair fixture was restored to its intentionally failing
state after evidence capture.

AUTONOMOUS GOAL COMPLETION: **4/4 (100%)**

VERIFIED GOAL COMPLETION: **3/4 (75%)**. DeepSeek's semantic explanation
completed but has no independent external correctness oracle; its read-only
project hash was still verified unchanged.

VALID ACTION RATE: **14/14 (100%)** on the final live sample. Deterministic safe
argument canonicalization repaired bounded `max_bytes`, factual absolute target
paths and factual project paths before validation.

CRITICAL VIOLATIONS: **0**

USER INTERVENTION RATE: **0/4 (0%)**

AVERAGE STEPS: **3.5** (median 2.5)

LOOP RATE: **0/4 (0%)**

AUTHORITY BLOCK RECOVERY: **1/1 (100%)**. A duplicate mutation proposal became
an observation; Qwen replanned to verification and completed.

UNNECESSARY DELEGATION: **0**

PREMATURE RESPONSE: **0**

PREMATURE MUTATION: **0**

EXPLICIT AGENT PRESERVATION: **100%**

READ_ONLY PRESERVATION: **100%**

FORBIDDEN AGENT PRESERVATION: **100%**

EXACTLY-ONCE: **preserved**. Live semantic fingerprints prevent repeated
effects; each approved NextAction receives its own PendingAction subturn, so a
post-mutation verification may intentionally rerun the same test while the same
action cannot execute twice.

SESSION AFFINITY: **preserved**. Delegation envelopes retain conversation/run
provenance, focused Codex thread affinity, original user text and agent binding.

## PERFORMANCE

PERFORMANCE BEFORE: **12–21 s per policy decision (approximate Phase 1.75
baseline)**

PERFORMANCE AFTER (5-call stable-state policy benchmark):

- p50: **8.557 s**
- p90: **13.215 s**
- p95: **13.215 s**
- mean: **9.492 s**
- input: **683 tokens/call**
- output: **123 tokens/call**
- compact payload: **1,886 bytes/call**
- model calls: **5/5**, no retry

The hot-path bottleneck is generation (~8.35 s/call). The first cold prefill was
~4.61 s; later same-prefix prefill was ~0.15 s. In larger real trajectories,
policy p50 ranged from 11.92 s to 15.98 s as factual observations increased the
context.

FAST PATH HIT RATE: **1/14 (7.14%)** in the final real sample; it emitted the
verified repair `RESPOND` without another model call.

DECISION CACHE HIT RATE: **0% live** because every sampled state changed;
**100% on the identical-state reuse regression**. Cache counters are per run.

INPUT TOKEN REDUCTION: **not numerically comparable** because Phase 1.75 did not
record a before-token baseline. The after prompt is measured at 683 input tokens
in the controlled benchmark.

MODEL CALL REDUCTION: **1 call (7.14%)** across the sampled live trajectory set
from the verified-completion fast path; **1 call (14.3%)** on the repair
trajectory alone. No second model/runtime was added.

## TEST AND SAFETY EVIDENCE

FULL TEST SUITE: **1218 passed, 1 skipped, 1 existing deprecation warning**

EXISTING REGRESSIONS: **0**

Focused authority/orchestration/capability/session/project/pending-action gate:
**298 passed**. Replay dataset: **14 scenarios**, zero real shadow effects,
explicit-agent and constraint preservation 100%, critical violations zero.

Real repair trajectory:

```text
INSPECT project
→ RUN tests (failed observation)
→ READ source
→ WRITE authorized fix
→ duplicate WRITE blocked as observation
→ RUN tests (passed verification)
→ FAST-PATH RESPOND
```

External pytest verification: **1 passed**.

## FILES

FILES CREATED (Phase 1.75/1.75B working tree):

- `tern/orchestrator/orchestration_contracts.py`
- `tern/orchestrator/orchestration_state.py`
- `tern/orchestrator/orchestration_policy.py`
- `tern/orchestrator/orchestration_loop.py`
- `tern/orchestrator/orchestration_shadow.py`
- `tern/orchestrator/orchestration_live.py`
- `tern/orchestrator/orchestration_fast_path.py`
- `tern/orchestrator/orchestration_replay.py`
- `tern/orchestrator/user_goal.py`
- `validation/phase175b_eval.py`
- `validation/phase175b_fixture/`
- orchestration contract/state/policy/loop/live/replay tests and fixtures
- this report

FILES MODIFIED (principal integration surfaces):

- `.env.example`
- `tern/orchestrator/agent.py`
- `tern/orchestrator/config.py`
- `tern/orchestrator/execution_authority.py`
- `tern/orchestrator/autonomy_foundation.py`
- `tern/orchestrator/tools.py`
- Codex/DeepSeek delegation and session integration tests

The checkout already contained broad uncommitted Phase 1.75 work; unrelated web
and site edits were preserved and are not claimed by this report.

## REMAINING RISKS

- Qwen generation remains the dominant latency; real multi-step tasks can still
  spend roughly one minute or more in policy calls.
- The live evaluation sample is intentionally small (n=4); 100% completion is
  evidence for the bounded slice, not a population estimate.
- Semantic-only agent answers are not independently verifiable without a task-
  specific oracle.
- High-risk confirmation/resumption deserves a dedicated end-to-end UI test;
  no high-risk effect was executed in this phase.
- Cache usefulness for real polling workloads needs a longer asynchronous-job
  corpus; current live hit rate was zero because each state changed.

## RECOMMENDED NEXT PHASE

Run a larger shadow/live trajectory corpus focused on real repository repairs,
async Codex jobs and authority-block recovery. Optimize Qwen generation/runtime
from measured timing before considering a fast/slow second policy. Keep training
report-only: collect preferred/rejected trajectories, but do not perform live
LoRA/SFT/DPO or self-label promotion.

## CONCLUSION

The JARVIS can now receive a goal, observe the real environment, choose one next
action with Qwen, execute only through deterministic authority, incorporate
actual observations, recover from a block, verify a mutation and finish without
user intervention. The authority boundary, READ_ONLY, explicit/forbidden agent
semantics, PathPolicy, TOCTOU checks and exactly-once protections remain intact.

**APPROVED — BOUNDED AUTONOMY VERIFIED**
