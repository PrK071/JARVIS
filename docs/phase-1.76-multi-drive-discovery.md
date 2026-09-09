# Phase 1.76 — Multi-Drive Environment Discovery & Project Resolution

## Status

**PARTIAL** — bounded discovery and project-cache integration are implemented and
validated in fixtures. Full live C:/D: evaluation remains environment-dependent.

## Architecture

`ProjectRegistry` remains the known-project cache. On a registry miss it invokes
`ProjectDiscovery`, which performs a bounded, marker-based, read-only search over
configured discovery roots. A successful result promotes only the exact project
directory to the operational `PathPolicy`; the discovery root is never promoted.

```text
registry → alias/name → bounded discovery → identity validation
         → exact project scope → registry cache → WorldState/OrchestrationLoop
```

Discovery has no write, delete, arbitrary execution, Git mutation, installation,
or job-creation capability. It skips Windows/system/profile-secret folders,
builds, caches, virtual environments, `node_modules`, and similar exclusions;
symlinks are not traversed.

## Configuration

```env
PROJECT_DISCOVERY_ROOTS=C:\Projects;D:\Projects
PROJECT_DISCOVERY_EXCLUDES=Windows,Program Files,AppData,node_modules,.venv,dist,build
```

When roots are omitted, safe user project folders plus common `C:/` and `D:/`
project roots are considered. A bare drive root uses a lower traversal depth,
directory/time budgets, early candidate limits, and deterministic ranking.

## Contracts and behavior

- `ProjectCandidate` contains canonical path, markers, match reasons, score,
  discovery root, and confidence.
- `DiscoveryResult` reports `RESOLVED`, `AMBIGUOUS`, or `NOT_FOUND`, roots and
  directories checked, elapsed time, budget exhaustion, and non-fatal errors.
- `discover_project(reference)` is a structured control-plane tool.
- Ambiguous candidates are never silently selected.
- Cached exact project scopes are restored on restart; discovery roots are not.
- Existing explicit-agent, READ_ONLY, authority, PathPolicy, and bounded-live
  gates remain responsible for substantive work and mutation.

## Validation

- Project discovery and registry tests: **37 passed**.
- Bounded-live/orchestration authority gate: **112 passed**.
- Full-suite validation should be rerun after the final environment-specific
  C:/D: fixture evaluation.

## Remaining risks

- Real drive-root latency and AccessDenied behavior require evaluation on the
  target machine.
- Manifest identity extraction is intentionally conservative and limited to
  lightweight metadata.
- Network/UNC/removable roots remain opt-in and are not scanned by default.

## Next step

Run a controlled Kari fixture on separate C:/ and D:/ roots, measure first-run
discovery versus second-run registry latency, and then promote to `APPROVED`
only after wrong-project resolution remains zero and the full regression suite
is green.
