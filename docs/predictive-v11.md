# Predictive v11: secure simulation foundation

## Freeze and scope

Predictive Consultative v1 remains frozen at `6b288b84fed38ce0904f0535a1748472bae5a59d`.
No causal model, recovery budget, grounding rule, target selection, or scoring
weight changed.  Its quality readiness applies to the validated v10 synthetic
scope, not arbitrary Python repositories.

Docker is absent on the current Windows host.  Runtime self-tests and candidate
simulation were therefore not executed.  Unit fakes cannot promote runtime
readiness.  No installation, service enablement, image pull, build, dependency
installation, or host project execution is automatic.

## Artifact retention

Ten v10 raw reports occupied 55,257,013 checkout bytes.  Their complete bytes
were compressed into `.orchestrator/predictive-artifacts/v10/<sha256>.json.gz`
and decompressed/verified before replacing the tracked files with compact
reports.  Compressed copies total 4,951,543 bytes; summaries total approximately
639 KB.  The local archive is an ignored working copy, not the sole durable
record: all originals remain in the original Git commit and on origin/main.

Each compact report retains aggregate metrics, population definitions, case
indices, failures, and provenance.  `artifact_retention` contains both the raw
checkout hash and the historical Git blob hash; CRLF/LF differences are explicit.
To recover the canonical complete bytes, retrieve the recorded `source_commit`
and `source_path` with `git show`.  Verify against `git_blob_sha256`.  For the
local gzip, verify decompressed bytes against `raw_sha256`.

Future evaluation traces should be written to ignored artifact storage or a
separately managed CI artifact store.  Git retains compact summaries, manifests,
hashes, factor/seed definitions, and failure indices.  `scripts/compact_predictive_artifacts.py`
is an explicit maintenance operation that verifies archives against the
preservation commit.  Existing history is never rewritten.

Clean-clone verification exposed an untracked empty fixture directory and
platform-dependent historical seals.  `.gitkeep` now preserves the directory.
Fixture/adjudication seals use LF bytes and exclude only `.gitkeep`; original
worktree seals remain in `manifest.json:historical_worktree_seals` and historical
reports.  No fixture code, oracle answer, or label was changed.

The external shadow/profiling traces are compacted by the same policy.  Their
complete bytes are preserved at `db82d7e05d7677b4b7fd491970ad4e96e73f95f3`
and in `.orchestrator/predictive-artifacts/v11/`; the current Git reports retain
the selected hypotheses, target-root match result, phase timings and provenance.

## Sandbox contract

`SandboxProvider` remains unavailable until all critical capabilities pass.
`UnavailableSandboxProvider` never dispatches a host process.  Docker provider
uses a Linux Engine with seccomp and a trusted, locally prepared immutable image
configured by `JARVIS_SANDBOX_IMAGE=<repository>@sha256:<digest>`.
The digest pins content; an operator must still review and trust the image.

The host prerequisites are Docker Desktop with Linux containers/WSL2 (or an
equivalent accessible Linux Docker Engine), cgroup v2, seccomp, and an approved
local image containing Python, `/bin/sh`, `/usr/bin/env`, and `/bin/sleep`.
Image-declared volumes are rejected; healthchecks and automatic pulls are
disabled.  No project Dockerfile, installer, or dependency hook is invoked.

The container runs UID/GID 65534, no capabilities, no-new-privileges, default
seccomp verified through `/proc`, read-only root, network none, bounded memory,
CPU, PIDs, output, core dumps, and wall time.  The active project is never mounted:
only explicitly scoped UTF-8 files are copied to a separate disposable input,
mounted read-only, and copied again into container tmpfs `/work`.
The snapshot is capped at 32 files/2 MiB, rejects protected paths and links,
and excludes everything not explicitly selected.

The workload environment is reconstructed with `/usr/bin/env -i`; host tokens,
SSH/AWS variables, Docker endpoints, and image environment do not reach it.
Only LANG, LC_ALL, TZ and PYTHONHASHSEED may be supplied in addition to fixed
runtime variables.  An image override after verification is rejected.

`jarvis predictive-sandbox-check --json` reports each capability.  Controlled
probes check root/input write denial, tmpfs writes, loopback-only network plus
connection denial, environment names, private paths, traversal/symlinks,
non-root UID, zero effective capabilities, seccomp/no-new-privileges, cgroup
limits, PID exhaustion, memory termination, timeout/child cleanup, and confirmed
container absence.  Daemon errors never count as confirmed cleanup.  Failures
revoke readiness.  Functional tests reduce risks; they are not proof against
every container or kernel escape.  Docker/host patching and image supply-chain
review remain operator responsibilities.

Real integration is opt-in with `RUN_PREDICTIVE_DOCKER_TESTS=true`.  Missing Docker
is a skip in the normal unit suite and an unavailable result at the runtime gate.

## Patch proposals and observed contracts

`PatchProposal.from_report` accepts an eligible grounded candidate, reuses the
existing candidate policy and repair-target validation, and records a stable
DecisionReport hash, analyzed commit, root, strategy, evidence IDs, targets,
prediction rank/score, file allowlist, budget, diff and SHA-256.
The content-only Qwen adapter receives facts/snippets and no writer or tools;
its request/token counters are separate from consultative reasoning.

`jarvis predictive-patch-check --project <root> --proposal <json> --json` previews
a strict unified diff in memory.  The command does not grant provenance or
execution authority merely because an input JSON declares a candidate ID.
The runtime factory is the provenance-preserving path from DecisionReport.

Validation blocks absolute/traversal/Windows device paths, protected or sensitive
paths, symlinks/junctions, binary content, renames, new files, deletions, fuzzy
preimages, malformed hunks, edits outside the typed target, unrelated symbols,
and sibling parameter/actual-argument changes.  Call-site proposals derived
from the report preserve positional slot or keyword binding identity.
V1 rejects unsupported diff forms rather than
guessing.  Defaults allow two files and 60 added/removed lines; hard ceilings are
three files, 120 lines each, and 64 KiB diff.  Python syntax is parsed statically;
no imports or tests execute on the host.  Config targets require explicit lines.

`ObservedCandidateOutcome` is versioned and preserves prediction, commit,
report ID, patch hash, validation, before/after tests, regressions, observed
evidence, timings, timeout, sandbox violation, and infrastructure status.
NOT_TESTED/SANDBOX_UNAVAILABLE cannot claim performed tests.  OBSERVED requires
baseline and patched tests without infrastructure failure.  `compare_observed`
is a pure function tested with fakes; it requires matched test IDs and a FAIL→PASS
improvement.  Already-passing tests, infrastructure failures, and equivalent
observations do not create an observed winner.  Passing targeted tests is not
universal correctness evidence.  Both rankings remain intact and authorization
stays false with approval required.

No patch applier, candidate test runner, simulation endpoint, auto-merge,
auto-commit, authority controller or ToolRegistry was added.  Those execution
components remain conditional on a real successful sandbox self-test.

## External shadow study

`scripts/predictive_external_shadow.py` inspected Requests 2.25.0 at
`03957eb1c2b9a1e5e6d61f5e930d7c5ed7cfe853`, using independent ground truth from
[issue 5667](https://github.com/psf/requests/issues/5667) and
[merged PR 5673](https://github.com/psf/requests/pull/5673).
One real encoding defect was presented with two wording scenarios.  Source
files were hashed before and after; the repository was unchanged.  No Requests
module or tests were imported/executed.  There were two local Qwen calls.

The frozen engine selected an import SCC in both scenarios instead of the
missing JSON encoding fallback in `requests/utils.py::get_encoding_from_headers`:
0/2 target-root matches, representing one independent defect.  This exposes a
real limitation: a legitimate production import SCC can attract a false causal
diagnosis in larger codebases.  The study is not a population accuracy estimate,
is not mixed into v10, and caused no engine tuning.

Qwen times were 26.5 s and 25.0 s; full pipeline times were 38.3 s and 33.6 s.
The historical v10 holdout latency of 59.1 s remains operational backlog.
Sandbox setup/baseline/patched tests/cleanup latency is NOT_MEASURED on this host.

A separate one-scenario profiling pass preserves the original study output.
It used cProfile without changing the engine: index 1.10 s, retrieval selection
0.20 s, context 0.02 s, ledger 6.05 s, causal analysis 4.55 s, inclusive recovery
0.07 s, Qwen 16.66 s, scoring 0.0002 s, total 28.77 s.  Cumulative recovery can
overlap causal-analysis work; these phase measurements must not be blindly
summed.  The profiling pass reproduced the same incorrect SCC diagnosis.

## Gate

Validation: 1496 passed / 3 skipped in the full suite, including 255 predictive
passes / 2 predictive skips.  Docker integration is opt-in and unavailable here;
the other predictive skip is host symlink creation without privilege.  Tests
used a process-local Python 3.13 Windows ACL workaround for pytest temporary
directories; production code and host permission settings were not modified.

`PREDICTIVE_QUALITY_READY = true` within the validated v10 scope.

`SIMULATION_RUNTIME_READY = false`

`CANDIDATE_SIMULATION = BLOCKED`

Configure a reviewed local image and accessible Linux Docker Engine, then run
the self-test in the target environment.  Only after all critical checks pass
should 8–12 controlled fixtures be used for baseline-versus-patch simulation.
