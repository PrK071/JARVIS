# PHASE 1.75 — SHADOW ORCHESTRATION FOUNDATION

Data: 2026-08-31

Status: **APPROVED**

## Architecture before

`intent_semantics.py` produced semantic facts consumed by the live routing
policy; `decision_policy.py` combined context, routing, selection and live
decisions; `agent.py` contained the live tool loop; `ExecutionAuthority` and the
execution gate already separated permission from selection, but there was no
versioned Goal → WorldState → NextAction → Observation loop.

## Architecture after

```text
SemanticUnderstanding → UserGoal
                         │
WorldState ──────────────┤
                         ▼
              OrchestrationPolicy
                         │
                    NextAction
                         │
                 ShadowActionSink
                         │
              Observation / reducer
                         └──────→ replan
```

The Phase 1.75 path receives immutable data snapshots. `OrchestrationLoop` and
`ShadowActionSink` receive no `ToolRegistry`, executor, Codex/DeepSeek manager,
job store, session registry, filesystem handle or Git handle. The sink records
proposals and exposes effect counters that remain zero. The existing
`ExecutionAuthority` is unchanged and remains the sole live gate.

The observer integrated into `Supervisor` is opt-in through
`ORCHESTRATION_SHADOW_ENABLED=false`. When enabled, it performs one Qwen policy
step and logs it without replacing or delaying the live decision by default.
Multi-step execution is available in the isolated replay runner, where all
observations are fixtures/replays/synthetic records.

## Reused components

- `CapabilityBaseline`, `CapabilityProfileBuilder`, `Agent` and `Capability`;
- grounded eligibility and its reason codes;
- `ProjectSnapshot.compact()` and project/Git/test facts;
- existing job status vocabulary;
- `ExecutionMode` and existing execution-gate facts;
- `LlamaClient` structured `json_schema` output;
- existing action logger and live-vs-shadow observability.

No parallel capability registry, project scanner, job manager, session manager
or execution authority was created.

## New contracts

- `UserGoal` schema version `1`;
- `WorldState` schema version `1`;
- closed `NextAction` schema version `1` with `INSPECT`, `DELEGATE`, `EXECUTE`,
  `ASK_USER`, `WAIT`, `RESPOND` and `STOP`;
- factual `Observation` schema version `1`.

`UserGoal` preserves explicit, conditionally permitted and forbidden agents as
different facts. Semantic action never selects an executor. `REPAIR` keeps the
mutation requirement `UNKNOWN`; concrete delete/remove/clear/reset operations
retain their distinct semantic actions.

## Files created

- `tern/orchestrator/orchestration_contracts.py`
- `tern/orchestrator/user_goal.py`
- `tern/orchestrator/orchestration_state.py`
- `tern/orchestrator/orchestration_policy.py`
- `tern/orchestrator/orchestration_loop.py`
- `tern/orchestrator/orchestration_shadow.py`
- `tern/orchestrator/orchestration_replay.py`
- `tests/fixtures/shadow_orchestration_scenarios.json`
- six focused test modules for contracts, understanding, state, policy, loop and replay

## Files modified

- `tern/orchestrator/config.py`: opt-in flag and explicit budgets;
- `tern/orchestrator/agent.py`: fail-open observer call whose result is ignored by live routing;
- `.env.example`: documented Phase 1.75 controls;
- `tests/test_execution_gate_shadow.py`: live isolation and failure-containment proof.

## Safety isolation

```text
Live authority changed: NO
Shadow execution capability: NONE
Shadow real side effects: 0
Shadow live decision changes: 0
ExecutionAuthority modified: NO
Phase 2 authority modes added: NO
```

The deterministic validator checks field compatibility, known agents/tools,
availability, capability claims, read-only conflicts, forbidden agents and
explicit-agent preservation. Invalid proposals become factual BLOCKED shadow
observations and are never adapted into executable tool calls.

## Stop and context controls

All limits are configurable: max steps, observations, action history, context
items, repeated actions, identical observations and failures. The reducer alone
updates state, compacts bounded histories and rejects observations tied to the
wrong action. Termination covers goal completion, terminal block, budget,
failure count, repeated action, identical observation/no progress and A/B/A/B
loops.

## Replay and evaluation

The fixture contains the ten required single-step cases and four multi-step
cases. Expectations support acceptable alternatives, acceptable agents,
forbidden actions and critical violations rather than exact-match only.

Deterministic reference-policy replay (14 scenarios, 19 actions):

```text
valid_action_rate:                 100.00%
unsafe_action_rate:                 0.00%
goal_progress_rate:                92.86%
agent_capability_match:           100.00%
explicit_agent_preservation:      100.00%
constraint_preservation:          100.00%
unnecessary_delegation_rate:        0.00%
premature_response_rate:            0.00%
premature_mutation_rate:            0.00%
ask_user_when_inspection_possible:  0.00%
loop_rate:                          7.14%  (the intentional no-progress fixture)
steps_to_goal:                       1.36
critical_shadow_violations:             0
real_side_effects:                      0
```

Qwen policy replay was also exercised against all 14 scenarios. Across the
observed 20 one-step decisions it produced no critical violation and preserved
explicit-agent, forbidden-agent and read-only constraints. It had one invalid
availability proposal (Codex unavailable) before replanning to `ASK_USER`, and
one unnecessary request for user context where fixture evidence was already
sufficient. Approximate observed metrics:

```text
valid_action_rate:                 95.00%
unsafe_action_rate:                 0.00%
goal_progress_rate:                78.57%
agent_capability_match:           100.00%
explicit_agent_preservation:      100.00%
constraint_preservation:          100.00%
critical_shadow_violations:             0
real_side_effects:                      0
```

Policy inference was roughly 12–21 seconds per action on the current local
runtime. For this reason, live observation remains disabled by default and the
full multi-step evaluation remains an offline/lab workflow.

## Regression result

```text
Existing regressions: 0
Full suite: 1188 passed, 1 skipped, 1 warning
New tests: 32
```

Covered suites include execution authority, execution-gate shadow, semantic
grounding, explicit binding, Codex sessions/jobs/collaboration, capability and
eligibility, Task Requirement Grounding, Evidence Provenance and Project
Intelligence through the complete repository suite.

## Requested conclusions

```text
Live authority changed: NO
Shadow execution capability: NONE
Existing regressions: 0
Critical shadow violations: 0
Explicit-agent preservation: 100%
Read-only preservation: 100%
Forbidden-agent preservation: 100%
Generic mutation → Codex coupling: REMOVED in UserGoal/Phase 1.75 shadow
Legacy live routing: UNCHANGED
Authority: UNCHANGED
Phase 2: NOT IMPLEMENTED
Training / LoRA / self-label promotion: NOT IMPLEMENTED
```

The historical live routing heuristic remains untouched solely to satisfy the
zero-regression/authority-preservation requirement. It is not used as the
semantic source of executor truth by the Phase 1.75 contracts or loop.
