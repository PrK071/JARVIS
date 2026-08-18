# Bonsai 27B runtime experiment — final record (closed)

Closed: 2026-08-16 (America/Sao_Paulo). Supersedes the intermediate checkpoint
`bonsai-runtime-experiment-checkpoint.md`, whose evidence is preserved in full below.

## Decision

```text
Bonsai 27B (Ternary-Bonsai-27B-Q2_g64)

Runtime compatibility       PASS
Structured output contract  PASS   (9/9 valid, no retry, no fallback)
Performance current HW      FAIL   (~0.27 tok/s)

reason:   operational throughput / hardware fit on RX 580 8 GiB + 16 GiB RAM
result:   not used for delivery; artifacts removed; primary local model stays Qwen3.5-4B
```

No further model exploration in this delivery. Do not re-download or re-test Bonsai.

## Candidate identity (official audit)

- Model repository revision: `abbae723028d71be674e71e1a71201a6f43fab22`
- Artifact: `Ternary-Bonsai-27B-Q2_g64.gguf`
- Size: `7,585,330,240` bytes
- SHA-256: `d30d09084082aa0f8e83d5d1bb8463089684f65281aac0785510f756622adda2`
- Group-64 is the mainline llama.cpp format (CPU, Metal, Vulkan, CUDA) and was the
  correct AMD/Vulkan candidate. Group-128 `Q2_0.gguf` requires the Prism fork and
  must not load in mainline. `PQ2_0` is experimental/unstable and was rejected.

GGUF header facts read directly from the file (no name inference):

```text
file_size_bytes              7585330240
gguf_version                 3
tensor_count                 851
general.architecture         qwen35
general.name                 Bonsai-27B
general.size_label           27B
general.file_type            41
general.quantization_version 2
qwen35.block_count           64
qwen35.ssm.state_size        128
qwen35.ssm.group_count       16
qwen35.ssm.inner_size        6144
tensor_type_histogram        F32 353, UNKNOWN_42 498
```

The 498 `UNKNOWN_42` tensors are the ternary Q2_0 planes that stock GGUF tooling
cannot name, consistent with the mainline group-64 format.

## Runtimes used

- Control: llama.cpp b10173 Vulkan, `10173 (e9fa0781f)`, Clang 20.1.8, Windows x86_64
- Candidate: llama.cpp b10437 Vulkan, commit `16d222fc5`
  - archive `llama-b10437-bin-win-vulkan-x64.zip`, `34,580,020` bytes
  - archive SHA-256 `50b58afc11c7c594cc131188e7cf76232f7c9744062df95282c3cc02d8efc405`
- Server arguments in effect: `-c 16384 -np 1 -fa on -ctk q8_0 -ctv q8_0 -ngl 99
  -fit on -fitt 1280 --host 127.0.0.1 --port 8080 -to 180 --jinja --reasoning off
  --no-ui --cors-origins localhost --metrics --slots`
- Endpoint `http://127.0.0.1:8080`, health `ok`, context `16384`, parallel slots `1`.
  Default server temperature `0.8`; callers must always send their frozen
  benchmark temperature explicitly.

## Hardware of record

- CPU: Intel Core i5-7400, 4 cores / 4 threads, 3.0 GHz
- RAM: `17,134,899,200` bytes (about 15.96 GiB)
- GPU: AMD Radeon RX 580 2048SP, driver 31.0.21925.1001
- VRAM: `8,589,934,592` bytes (8 GiB), verified via registry
  `HardwareInformation.qwMemorySize`. Any earlier report stating 4 GiB was a
  `Win32_VideoController.AdapterRAM` signed-32-bit saturation artifact and must be
  read as 8 GiB.
- Secondary GPU present but unused by the Vulkan runtime: Intel HD Graphics 630

## Bonsai measured results

Structured contract on b10437 (`context_size` 8192, `model_type` `Q2_0`):

- 9/9 valid, validity `1.0`, empty failure taxonomy, no retry, no fallback
- latency ms: mean `147,991.42`; p50 `104,357.51`; p90 `228,323.95`;
  p95 `352,721.60`; p99 `452,239.71`
- tokens: 449 prompt / 290 completion

Performance, GPU 24 layers offloaded (1 request, 32 prompt / 64 completion):

- TTFT `104,766.55` ms; total latency `342,690.71` ms
- throughput `0.26899` tok/s; finish reason `length`
- peak working set `7,333,761,024` bytes; peak private `5,261,725,696` bytes
- mean CPU `37.77%` over 2,723 samples

Performance, CPU-only (2 requests, 32 prompt / 64 completion each):

- TTFT mean `63,909.63` ms; latency mean `287,879.96` ms
- throughput mean `0.28582` tok/s (`0.29034`, `0.28131`)
- peak working set `9,985,724,416` bytes; peak private `5,019,697,152` bytes
- mean CPU `58.32%` over 4,922 samples

Conclusion: the contract layer worked perfectly; throughput was two orders of
magnitude below usable (`~0.27` tok/s versus Qwen3.5-4B seconds-scale responses),
so migration was rejected on hardware fit, not on correctness.

## Prior attempt failure mode (b10173, superseded by the b10437 run)

An earlier Bonsai attempt on b10173 failed every constrained call because the
model's reasoning delimiters conflicted with grammar-constrained output. Decisive
server line:

```text
error: Failed to initialize samplers: Unexpected empty grammar stack after accepting piece: <think>
```

Its autonomy dry-run artifact recorded 12 cases, `dry_run: true`,
`execution_count: 0`, `json_validity: 0.0`, `retry_rate: 1.0`, `fallback_rate: 0.0`,
0 prompt/generated tokens, latency p50 `87.32` ms — that is, the requests never
produced output. Safety invariants still held in that failure:
`no_execution_from_dry_run_accuracy: 1.0`, `false_eligible_agent_rate: 0.0`,
`capability_profile_accuracy: 1.0`, `availability_separation_accuracy: 1.0`.
The b10437 run above replaced this failure mode with full 9/9 structured validity,
which is why the final rejection is throughput, not contract.

## Qwen3.5-4B controls captured (baseline preserved)

Model `D:\JARVIS\models\Qwen_Qwen3.5-4B-Q4_K_M.gguf`, `3,013,027,808` bytes,
`/props` model type `Q4_K - Medium`.

Progressive contract on b10173 (`b10173-e9fa0781f`):

- 9/9 valid, no retry/fallback
- latency ms: mean `3,063.34`; p50 `1,506.26`; p90 `5,573.74`; p95 `9,725.32`; p99 `13,046.59`
- tokens: 449 prompt / 270 completion
- artifact `.orchestrator/model-contract-qwen-b10173.json`

Progressive contract on b10437:

- load/switch time observed `35.5` s
- 9/9 valid, no retry/fallback
- latency ms: mean `3,894.04`; p50 `1,640.42`; p90 `8,001.11`; p95 `13,633.64`; p99 `18,139.67`
- tokens: 449 prompt / 274 completion
- artifact `.orchestrator/model-contract-qwen-b10437.json`

Agentic reasoning diagnostic on b10437 (dry-run, zero automatic actions):

- 6/6 structurally valid; automatic actions `0`
- factor precision/recall/F1: `54.58%` / `94.44%` / `68.35%`
- step recall `8.33%`; ordering accuracy `5.56%`
- verification recall `80.56%`; uncertainty recall `83.33%`
- forbidden-factor accuracy `100%`
- mean latency `41,097.19` ms; tokens 1,195 prompt / 2,600 completion
- artifact `.orchestrator/agentic-reasoning-qwen-b10437-final.json`
- A separate 320-token pilot is retained but excluded from A/B: four truncations
  plus two validator failures exposed that llama.cpp does not enforce
  `uniqueItems` in constrained arrays. The final diagnostic uses closed boolean
  maps (no output repair) and 512 tokens identically for both models.

These Qwen numbers are the standing reference for planning/decomposition work and
show where the 4B is weak (step recall, ordering) without any model change.

## Generic infrastructure produced by the experiment (kept)

- `tern/orchestrator/local_model_runtime.py`: one model-agnostic contract with
  `health`, `model_info`, `runtime_info`, `generate_text`, `generate_structured`;
  strict `json.loads`, JSON Schema Draft 2020-12 validation, optional semantic
  validator, failure taxonomy, prompt-free inference telemetry.
- `tern/orchestrator/local_model_contract_eval.py`: nine progressive schemas from
  trivial object to grounded requirements and the current semantic schema.
- `tern/orchestrator/local_model_reasoning_eval.py` and
  `local_model_performance_eval.py`: dry-run reasoning diagnostic and
  TTFT/throughput/resource measurement, both model-independent.
- `scripts/gguf_metadata.py`: reads GGUF header metadata without loading tensors.
- Config aliases `LOCAL_MODEL_PROVIDER`, `LOCAL_MODEL_PATH`, `LOCAL_MODEL_RUNTIME`;
  legacy `MODEL_*` variables remain compatible.
- `client.py`: `ServerTimeoutError` / `ServerUnavailableError` taxonomy and SSE
  streaming used for TTFT measurement.
- `RuntimeManager` now reports a different executable as `runtime_executable`
  mismatch. Before the fix it considered b10173 and b10437 interchangeable when
  model and arguments matched.

## Frozen foundations during the experiment

Explicit Agent Binding, Codex Session Affinity, Autonomy Foundation, Task
Requirement Grounding + Evidence Provenance, Project Intelligence V2. Automatic
agent selection, delegation, mutation, task decomposition, planning, verification
loops, fallback switching and replanning remained disabled and remain disabled.

## Repository state at experiment time

- Repository `C:\Users\User\JARVIS`
- Baseline commit `dedeaceff663315b280d73561e4aa5b1743bee37` (`feat: add project
  intelligence v2`), clean before the experiment
- Experiment branch `feat/bonsai-runtime-migration-gate` carried no commits of its
  own; all work stayed in the working tree
