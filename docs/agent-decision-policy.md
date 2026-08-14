# Agent Decision Policy

The Qwen model remains the orchestrator. `AgentDecisionPolicy` is deterministic
scaffolding: it supplies a compact local context, a recommended intent, a bounded
tool plan, observable reason codes, and side-effect metadata. Qwen still writes
tool arguments and the existing confirmation/security layer still authorizes and
executes calls.

## Previous decision path

```text
user transcript
-> technical-name/STT normalization
-> ProjectRegistry context
-> scattered Codex/DeepSeek/history/project gates in Supervisor
-> full or filtered tool catalogue
-> Qwen tool choice
-> PendingActionStore and PathPolicy
-> ToolProgressTracker
-> tool execution/result
-> Qwen answer
```

The special cases were split between the system prompt, `agent.py`, transcript
normalization and tool descriptions. There was no cross-turn entity focus, no
structured confidence/reason trace, and no deterministic corpus for measuring
status/history/delegation confusion.

## Current decision path

```text
original transcript
-> contextual technical-name normalization
-> compact DecisionContext from already-known local state
-> AgentDecisionPolicy recommendation
-> Qwen chooses an answer or supplied tool and arguments
-> existing confirmation/security policy
-> ToolProgressTracker
-> execution and structured result
-> ConversationFocus update
-> Qwen answer
```

`DecisionContext` does not call Codex, DeepSeek or another paid/remote model. It
contains the active project, known job/thread/session state, recent real tool
results, pending action and focused agent/project/file/job/session.

## Invariants

- Codex status uses `get_codex_job_status`; history uses
  `review_codex_session`; neither creates a turn.
- New project work uses `delegate_to_codex`.
- DeepSeek is delegated only on an explicit request while automatic escalation
  remains disabled. Reviewing its persisted session never calls its API.
- A follow-up uses the focused entity when there is one plausible antecedent.
- A direct answer receives no tools when the policy has strong confidence that
  existing context is sufficient.
- Existing `PendingActionStore`, `PathPolicy` and confirmations remain after the
  decision; the policy does not bypass security.
- A repeated call without progress receives an actionable reuse/replan message.

## Side effects

Every registered tool is classified as one of `READ_ONLY`, `LOCAL_MUTATION`,
`REMOTE_READ`, `REMOTE_GENERATION`, or `CODE_EXECUTION`. The classification is
diagnostic and does not replace the security policy.

## Diagnostics

Dry-run a single decision without executing a tool:

```powershell
python -m tern.orchestrator agent-decision "o codex terminou?"
```

Run the frozen baseline and current deterministic benchmark:

```powershell
python -m tern.orchestrator agent-routing-eval
python -m tern.orchestrator agent-routing-eval --split holdout --confusion
```

The optional live mode starts/reuses local Qwen, exposes mocked/dry-run tool
schemas and records choice and latency. It never dispatches a selected tool:

```powershell
python -m tern.orchestrator agent-routing-eval --live-qwen --split holdout
```

The normal benchmark reads `tests/data/agent_routing_cases.jsonl`; it performs no
remote calls, filesystem mutations, Codex turns or DeepSeek requests.

## Evaluation protocol after phase 1

The original 100-case corpus is named `agent_routing_regression`. Its original
80/20 split is retained only as history: the first sealed holdout scored 95%, but
case V010 was inspected and the policy was then changed. Consequently:

```text
sealed_holdout_first_pass = 95%
holdout_contaminated_after_case_V010_analysis = true
regression_score_after_fix = 100%
```

It is incorrect to present the later 100% as independent generalization. The
100 cases are now a required 100/100 regression gate.

Independent evaluations use versioned files such as
`agent_routing_test_v2.jsonl`. A test set is created and hashed before policy
work, evaluated exactly once at the end, and never edited from its result. After
opening its result it becomes known regression data. A future independent
measurement must create `test_v3`, then `test_v4`, and so on.

The sealed v2 valid pass on 2026-08-08 scored 23/40 (57.5%). Its 17 failures
were recorded without changing the policy. The initial harness attempt aborted
because the reporter incorrectly required a development/holdout `split`; only
that optional-field access was repaired before the valid pass. The case-file
checksum remained unchanged. V2 must not be presented as future holdout data.

## Shadow observation

Set `AGENT_DECISION_SHADOW=true` to append compact redacted events to
`.orchestrator/agent-decisions.jsonl`. Shadow mode never changes the decision or
executes an additional action. Events contain intent, confidence, reason code,
focused entity identifiers, planned/actual tool names, prompt-size estimates,
timings and one outcome (`success`, `tool_error`, `user_cancelled`, `clarified`,
`loop_prevented`, `delegated`, or `direct_answer`). File contents and tool result
bodies are not copied into this log. API keys, bearer values and credential-like
fields use the existing `ActionLogger` redaction.

Human feedback is annotation only; there is no online learning:

```powershell
python -m tern.orchestrator agent-feedback --last wrong --expected CODEX_REVIEW
python -m tern.orchestrator agent-feedback --last expected=CODEX_REVIEW
python -m tern.orchestrator agent-decision-stats --days 7
```

## Timing and safe fast paths

Supervisor results and shadow events separate context, policy, prompt, Qwen,
tool and response timings. The current Qwen client is non-streaming, so
`qwen_first_token_ms` is explicitly `null` with `qwen_streaming=false`; the
system does not claim an invented first-token measurement. Voice sessions add
`time_to_first_audio_ms` when TTS actually starts.

`AGENT_DECISION_FAST_PATH=true` may pre-dispatch a high-confidence,
single-entity `READ_ONLY` operation: one active Codex job status, one shared
Codex thread review, one known DeepSeek session review, or one focused file
read. An unambiguous explicit user handoff to Codex or DeepSeek also bypasses
semantic and argument-generation inference: the original request becomes the
task and the active project becomes the project path. Arguments still pass
through normal tool validation, project policy and confirmation controls. Other
generation, cancellation and mutation routes remain excluded.

The small decision-context cache is event-invalidated on project, job, file,
session and pending-action changes; it does not poll external services.

## Phase 3 semantic representation

Phase 3 inserts an auditable representation before the final intent:

```text
normalized input
-> IntentFrame (speech act, operation, execution_requested, constraints)
-> ConversationReferenceResolver (typed candidates, recency and verb compatibility)
-> AgentDecisionPolicy tool plan
-> semantic constraint validation
-> existing security and execution layers
```

`IntentFrame` distinguishes questions from commands and records
`QUESTION`, `REQUEST`, `COMMAND`, `EXPLANATION_REQUEST`, `STATUS_QUERY`,
`REFERENCE_QUERY`, `CONFIRMATION`, and `CORRECTION`. Constraints include agent
forbids, mutation/new-turn/cancel/delegation forbids, self-answer, read-only,
background/wait, and ordered plans. A mention of an agent does not itself request
execution. When `execution_requested=false`, only read-only tools may be selected.

The reference resolver scores the last ten typed entities by explicit mention,
verb/type compatibility, state and turn-distance decay. A running job remains a
strong candidate for “terminou?”, while a file is favored by “abre”. Ties between
plausible types produce clarification. `ConversationFocus` keeps this short stack;
it is updated only from real decisions/results, not guessed long-term memory.

Before dispatch, `constraint_violation_for_tool` independently rejects a tool that
contradicts the frame. This is intent semantics, not a replacement for `PathPolicy`,
pending actions, confirmations, or other security checks. Shadow events now include
speech act, `execution_requested`, constraints, reference type/confidence and
follow-up type. Structured dry-run diagnostics are available with:

```powershell
python -m tern.orchestrator agent-decision "não cancela, só vê se terminou" --explain
```

## Phase 3 evaluation

The 17 v2 failures were captured before changes in
`agent_routing_v2_failure_taxonomy.json`. Their primary causes were: reference
resolution (5), negation scope (3), action requests (3), temporal follow-up (3),
question-about-action (1), result reuse (1), and compound action (1).

Known regression results after the semantic work:

- original corpus: 100/100;
- known v2: 40/40;
- semantic regression: 64/64, including semantic-field/property checks;
- known-difficult live Qwen dry-run: 19/20 (95%), zero real tools, turns, API
  calls or loops; decision p50/p90/p95 9.324/25.198/27.244 seconds.

The independent v3 was authored only after implementation, regressions, live
dry-run and the full suite were frozen. It contains 50 cases in five equal
categories, 19 adversarial cases, 10 multi-turn scenarios and 10 STT/noisy cases.
Its sealed SHA-256 is
`382fbfde9951ae3e9c5a61160c2c7a16bfe641cccd0f5170a5826d38b91d516e`.

The one valid v3 pass scored 17/50 (34% overall), 56% intent/tool selection,
96% project, 16.7% constraint satisfaction and 87.5% reference resolution, with
zero Codex new-turn violations and zero loops. No policy change was made from
these failures. V3 is now known data; a future independent measurement must use
v4. The sealed metadata and immutable result live beside the corpus in
`tests/data`.

## Phase 4 semantic-first implementation

`QwenSemanticInterpreter` is an isolated, zero-tool semantic pass. It receives
original and normalized input plus compact `DecisionContext`, focus and recent
entity types. It returns only strict JSON validated into `SemanticDecision`:
speech act, existing `Intent`, operation, execution flag, agent, semantic target,
existing constraints/follow-up type, ordered compound steps, ambiguity and
semantic confidence. Concrete paths, IDs, thread IDs and session IDs are rejected.

The pass has one repair attempt containing only schema error, invalid object and
schema. A second failure uses a safe fallback: unequivocal reads remain possible;
generation, mutation, cancellation and steering never run. The semantic cache key
includes message plus project, focus, job, session and pending-action identity.
The selector skips simple knowledge questions but is conservative for agent names,
references, negation, follow-ups, execution and compound language.

Policy consumes the validated frame, then resolves its semantic target to a real
reference. It keeps semantic and reference confidence separate. Every catalog and
dispatch is independently checked against explicit constraints. When
`execution_requested=false`, only `READ_ONLY` tools can remain available.

Compound steps are exposed one at a time. `positive_recommendation` is accepted
only after a DeepSeek step returns explicit boolean
`positive_recommendation`; missing or false evidence blocks the Codex step and
asks for clarification. This prevents a conditional plan from becoming an
unconditional delegation.

The known semantic regression v2 has 100 cases: all 33 v3 failures plus 67
variations, including 30 minimal pairs. Property tests enumerate every registered
tool for `FORBID_CODEX`, `FORBID_DEEPSEEK`, `FORBID_CANCEL`,
`FORBID_NEW_TURN`, `FORBID_DELEGATION`, `READ_ONLY`, `ANSWER_SELF` and
`execution_requested=false`.

V4 is intentionally not created or sealed yet. The local Qwen/llama-server
completed the corrected one-case dry-run in 19.1 seconds, but repeated 30-case and
five-case dry-runs exceeded the configured 180-second request timeout. Complete
live-Qwen, semantic-first A/B and latency measurements are prerequisites for V4;
no corpus is created until they complete.
