---
name: git-checkpoint-push
description: Create a safe JARVIS checkpoint commit and push it to origin/main when explicitly invoked or when a configured session timer is near its deadline.
metadata:
  short-description: Safe JARVIS checkpoint and push
---

# Git Checkpoint Push

Use this skill only after an explicit `$git-checkpoint-push` request, a
`/git-checkpoint-push` command, or a configured JARVIS checkpoint timer. It
creates a recovery checkpoint for `github.com/PrK071/JARVIS`; it is not a
general-purpose git automation skill.

## Time limit

Codex does not expose a reliable remaining five-hour quota to tools. Do not
claim that an exact quota boundary is observable. For an elapsed-time
checkpoint, start this watchdog at the beginning of a session:

```powershell
jarvis git-checkpoint-watch --session-seconds 18000 --lead-seconds 90
```

It attempts the checkpoint 90 seconds before five elapsed hours. Keep that
terminal alive. For an immediate checkpoint, invoke the slash command in a
JARVIS text session or run:

```powershell
jarvis git-checkpoint-push --message "chore: checkpoint before session limit"
```

## Safety contract

The command must fail closed. Before mutation, verify all of the following:

- repository root is the intended JARVIS root;
- current branch is `main`;
- remote `origin` resolves to `github.com/PrK071/JARVIS`;
- the index has no pre-existing staged changes;
- modified and untracked paths exclude `.git`, `.env*`, `_arquivo/`, `runtime/`,
  `models/`, `*.gguf`, and `interface/providers.json`;
- `git diff --check` passes.

Never force-push, pull/rebase/reset, amend a commit, change git configuration,
or include protected paths. If a check fails, report the blocker and leave the
working tree and index as they are.

`--run-tests` is optional. Use it when time permits; a deadline checkpoint may
use the default fast preflight so it can preserve work before the session ends.

## Codex operation

When invoked in Codex, use the JARVIS CLI instead of reconstructing git steps:

```powershell
jarvis git-checkpoint-push --message "chore: checkpoint before session limit"
```

Use `--dry-run` first when the repository state is unknown. After a successful
checkpoint, report the commit SHA, push result, and any skipped verification.

## JARVIS text command

In `jarvis text`, the explicit command is:

```text
/git-checkpoint-push chore: checkpoint before session limit
```

This command bypasses the Qwen tool registry and runs only because the user
typed the slash command. It does not grant Qwen autonomous git authority.
