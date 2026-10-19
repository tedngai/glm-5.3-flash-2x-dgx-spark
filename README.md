# GLM-5.3-Flash on 2 × DGX Spark

Run **GLM-5.3-Flash** on one pair of NVIDIA DGX Sparks (GB10, 128 GB unified
memory each) connected by their ConnectX-7 ports. The measured recipe serves
NVIDIA's pre-quantized **FP4-Local** checkpoint through **vLLM/B12X**, with
tensor parallelism across the two Sparks, FP8 KV cache, MTP speculative
decoding, eight admission slots and a **500,000-token** served window. Text,
JSON-schema output, tool calls, single-image vision and the Anthropic Messages
surface are present. This is an experimental single-pair recipe, not a
turnkey distribution.

The repository contains the pinned launcher, the exact recipe, a reproducible
benchmark harness and the measured result set. Successful loading, measured
speed, populated context capacity and feature correctness are reported as
separate milestones.

## 1. Current status

| Item | Verified result or current limit |
| --- | --- |
| Model precision | NVIDIA `GLM-5.3-Flash-FP4-Local`: NVFP4 routed experts, MXFP8/FP8 and BF16 elsewhere, FP8 KV cache. No local requantization. |
| Hardware | 2 × DGX Spark GB10, 128 GB unified memory each, ConnectX-7 RoCE link, TP2 across the pair. |
| Engine | vLLM `0.1.dev21830+g8244d1c44` in `eugr/spark-vllm-b12x@sha256:03a956…3595`, B12X attention/MoE/linear/loader. |
| API model ID | `nvidia/GLM-5.3-Flash-FP4-Local` |
| Served context | **500,000 tokens** per request. The model's native window is 1,048,576; it is not enabled in this recipe. |
| KV capacity | **1,617,857 tokens** at the measured start. Three 500k windows fit in principle; the matrix populated 1.2M tokens (6 × 200k, 3 × 400k). Eight 500k windows do not fit. |
| Admission slots | `max-num-seqs 8`. The matrix records actual overlap; a requested C8 is not counted as eight streams without it. |
| Speculative decoding | MTP with **one** draft token, `moe_backend=humming`, `attention_backend=B12X`. |
| Speed | 37-cell matrix in §4: 32.6–37.1 tok/s total decode for one stream; 86–146 tok/s at C8 depending on output budget; 2,150–2,440 tok/s prefill from 8k–400k. |
| Long context | 400k row completed (C1/C2/C3, 181–373 s TTFT); 190k needle correct (190,012 prompt tokens); 496k single request passed at 230.7 s TTFT. |
| Text, JSON, tools | Arithmetic, schema-constrained JSON and a tool-call/result round trip passed. |
| Vision | Single-image red-square check passed. Multi-image ordering, video and tool-result media are not qualified. |
| Anthropic Messages | `messages` and `count_tokens` pass. The 400 error envelope and chunked request bodies differ from Anthropic's shape. |
| Thinking control | The shipped chat template always opens a thinking block; `enable_thinking:false` is a no-op. `reasoning_effort` `low`/`high`/`max` is the available control. |
| Reliability | No one-hour soak and no rollback exercise yet. |

**Scope:** the publishable claim is the measured FP4/vLLM recipe on this pair.
The separate TensorFold EXL3 engine is retained only as A/B history in
[`EVAL-REPORT.md`](EVAL-REPORT.md); its numbers are not merged into the FP4
matrix.

## 2. Start the service

Requirements: two DGX Sparks with ~110 GiB free unified memory each, a direct
ConnectX-7 link with working RoCE v2, Docker with the NVIDIA runtime, key-based
SSH from the head to the worker, and ~180 GB of model/cache headroom per node.
The launcher discovers the pair and starts one rank on each Spark.

```bash
git clone https://github.com/tedngai/glm-5.3-flash-2x-dgx-spark.git
cd glm-5.3-flash-2x-dgx-spark
./start.sh --setup          # first run: image + model, then start
```

On later boots:

```bash
./start.sh                  # background; prints the launcher log path
DAEMON=0 ./start.sh         # foreground
```

`start.sh` uses `spark-vllm-docker` at `0bdc9c9` and the tuned recipe in
[`recipe/glm-5.3-flash-tuned.yaml`](recipe/glm-5.3-flash-tuned.yaml). Set
`SPARK_VLLM_DIR` to reuse an existing checkout. The launcher builds/pulls
`vllm-node-b12x`, downloads or copies the checkpoint, starts TP2 serving and
runs its own startup checks. If the ConnectX-7 addresses changed, refresh the
launcher's saved cluster with
`./run-recipe.sh --discover -n <head-ip>,<worker-ip>` once. `start.sh` refuses
to start while another process answers on port 8000; stop the old service
first.

The API binds to port **8000** on the head. There is no authentication in this
recipe; keep it on a trusted network.

```bash
curl -s http://127.0.0.1:8000/health
curl -s http://127.0.0.1:8000/v1/models
curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"nvidia/GLM-5.3-Flash-FP4-Local",
       "messages":[{"role":"user","content":"What is 19 + 23?"}],
       "max_tokens":128,
       "chat_template_kwargs":{"reasoning_effort":"low"}}'
```

Stop both ranks with:

```bash
./stop.sh
```

The model always thinks before answering. Give chat requests enough
`max_tokens`; a 64-token budget can be consumed by the reasoning block and
return no final content.

## 3. Configuration

The published profile is the tuned recipe, not the upstream stock defaults.

| Setting | Published value | Stock upstream | Why |
| --- | --- | --- | --- |
| `max-num-seqs` | **8** | 4 | Uses the 1.62M-token KV pool for multi-user bursts. |
| `max-num-batched-tokens` | **8192** | 4096 | Improves multi-stream decode without oversubscribing the pair. |
| `num_speculative_tokens` | **1** | 5 | Raised draft acceptance and step rate; see §5. |
| `max-model-len` | **500000** | 500000 | Fits the checkpoint and leaves host memory headroom. Native 1M is not enabled. |
| `gpu-memory-utilization` | **0.87** | 0.87 | The measured stable allocation. |
| KV cache | FP8 | FP8 | `--kv-cache-dtype fp8`. |
| Attention / MoE / loader | B12X | B12X | `--attention-backend B12X`, `--moe-backend b12x`, `--load-format b12x`. |
| Prefix caching | on | on | `--enable-prefix-caching`. |
| Chunked prefill | on | on | `--enable-chunked-prefill`. |
| MTP backends | humming + B12X | humming + B12X | `{"method":"mtp","num_speculative_tokens":1,...}`. |

The recipe file is the executable source of truth. Three values differ from
upstream: `max_num_seqs`, `max_num_batched_tokens` and
`num_speculative_tokens`. The launcher's default recipe is intentionally not
edited in place; the tuned copy is passed explicitly.

**Capacity is not the same as context.** The KV pool holds 1,617,857 tokens.
A single request can use up to 500,000; eight requests cannot each use 500,000.
At 200k inputs the measured recipe admits six (1.2M prompt tokens); at 400k it
admits three (1.2M prompt tokens). The matrix skips cells above 1.55M
aggregate prompt tokens rather than allowing the scheduler to queue or retract
them, so 8 × 200k and 4 × 400k are intentionally absent.

**Thinking-mode caveat.** The checkpoint's chat template opens a thinking
block and offers only `reasoning_effort` `low`/`high`/`max`. The TensorFold
engine supports a true thinking-off mode; this vLLM recipe does not. Any chat
decode comparison across the two stacks is therefore not apples-to-apples
until a modified template is supplied.

## 4. Current local speed sweep

The sweep uses exact-length synthetic prompts, a unique nonce prefix per
request (no prefix-cache reuse), 2,048 forced output tokens and the highest
concurrency that fits the measured KV pool. Every request completed with the
requested token count. These are **performance tests, not quality scores**.
The requests use raw completions rather than the chat template so prompt and
output lengths are exact; the chat-based A/B numbers in §5 are therefore not
directly comparable to this table.

**Measured matrix — 37 cells, 2,048 forced output tokens.** All 37 cells
completed with the requested overlap and no errors. Total decode is 32–36
tok/s for one stream, 50–62 tok/s at C2, 67–86 tok/s at C4, 64–80 tok/s at C6
and 86–116 tok/s at C8. The C6 cells and 400k/C3 are not monotonic with the
row above; because total decode is measured inside the common window, a more
staggered wave can report a lower aggregate. Treat those cells as load-shape
observations, not a clean concurrency-scaling curve. Prefill stays at 2,150–2,440 tok/s from
8,192 to 400,000 input tokens; TTFT scales with input length and concurrency.
No request was preempted (`vllm:num_preemptions_total` stayed at 0).

| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | Decode/request tok/s | TTFT s | Shared decode s | Peak overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 845.1 | 36.3 | 36.3 | 0.6 | 56.3 | 1 | 1/1 |
| 512 | 2 | 1348.1 | 62.0 | 31.0 | 0.8 | 66.0 | 2 | 2/2 |
| 512 | 4 | 1266.4 | 75.6 | 18.9 | 1.6 | 108.0 | 4 | 4/4 |
| 512 | 6 | 1551.8 | 79.5 | 13.4 | 2.0 | 152.2 | 6 | 6/6 |
| 512 | 8 | 1519.8 | 116.2 | 14.5 | 2.7 | 140.5 | 8 | 8/8 |
| 2,048 | 1 | 1784.3 | 35.5 | 35.5 | 1.1 | 57.7 | 1 | 1/1 |
| 2,048 | 2 | 1838.2 | 54.3 | 27.2 | 2.2 | 75.2 | 2 | 2/2 |
| 2,048 | 4 | 2395.2 | 86.1 | 21.5 | 3.4 | 95.0 | 4 | 4/4 |
| 2,048 | 6 | 2153.0 | 69.7 | 11.8 | 5.6 | 172.3 | 6 | 6/6 |
| 2,048 | 8 | 2234.4 | 96.4 | 12.2 | 7.3 | 167.2 | 8 | 8/8 |
| 8,192 | 1 | 2349.4 | 35.9 | 35.9 | 3.5 | 56.9 | 1 | 1/1 |
| 8,192 | 2 | 2318.4 | 51.4 | 25.7 | 7.0 | 77.5 | 2 | 2/2 |
| 8,192 | 4 | 2299.1 | 71.9 | 18.3 | 14.0 | 111.4 | 4 | 4/4 |
| 8,192 | 6 | 2286.1 | 70.5 | 11.9 | 19.2 | 170.7 | 6 | 6/6 |
| 8,192 | 8 | 2294.4 | 101.8 | 12.8 | 22.7 | 158.3 | 8 | 8/8 |
| 32,768 | 1 | 2434.5 | 34.6 | 34.6 | 13.5 | 59.2 | 1 | 1/1 |
| 32,768 | 2 | 2384.6 | 50.9 | 25.5 | 22.1 | 78.0 | 2 | 2/2 |
| 32,768 | 4 | 2361.1 | 70.4 | 17.9 | 44.4 | 112.7 | 4 | 4/4 |
| 32,768 | 6 | 2351.5 | 64.9 | 10.7 | 58.3 | 176.6 | 6 | 6/6 |
| 32,768 | 8 | 2344.3 | 96.7 | 12.4 | 72.3 | 161.5 | 8 | 8/8 |
| 65,536 | 1 | 2440.0 | 35.7 | 35.7 | 26.9 | 57.4 | 1 | 1/1 |
| 65,536 | 2 | 2386.8 | 50.5 | 25.2 | 42.6 | 79.7 | 2 | 2/2 |
| 65,536 | 4 | 2362.8 | 67.9 | 17.0 | 78.8 | 114.1 | 4 | 4/4 |
| 65,536 | 6 | 2350.7 | 65.5 | 10.9 | 106.7 | 176.8 | 6 | 6/6 |
| 65,536 | 8 | 2343.1 | 85.9 | 10.6 | 134.3 | 173.9 | 8 | 8/8 |
| 131,072 | 1 | 2365.5 | 35.3 | 35.3 | 55.4 | 57.9 | 1 | 1/1 |
| 131,072 | 2 | 2351.1 | 52.5 | 26.2 | 84.9 | 76.5 | 2 | 2/2 |
| 131,072 | 4 | 2323.6 | 67.1 | 16.8 | 150.0 | 112.0 | 4 | 4/4 |
| 131,072 | 6 | 2313.2 | 64.3 | 10.7 | 206.7 | 176.4 | 6 | 6/6 |
| 131,072 | 8 | 2298.0 | 91.5 | 11.7 | 263.8 | 155.2 | 8 | 8/8 |
| 200,000 | 1 | 2353.0 | 35.4 | 35.4 | 85.0 | 57.8 | 1 | 1/1 |
| 200,000 | 2 | 2308.5 | 50.0 | 25.0 | 131.9 | 79.3 | 2 | 2/2 |
| 200,000 | 4 | 2286.6 | 71.4 | 17.9 | 223.5 | 103.4 | 4 | 4/4 |
| 200,000 | 6 | 2268.8 | 67.2 | 11.4 | 312.4 | 157.7 | 6 | 6/6 |
| 400,000 | 1 | 2204.8 | 32.6 | 32.6 | 181.4 | 62.7 | 1 | 1/1 |
| 400,000 | 2 | 2176.2 | 51.9 | 26.0 | 276.4 | 75.0 | 2 | 2/2 |
| 400,000 | 3 | 2156.9 | 48.5 | 16.6 | 373.4 | 111.3 | 3 | 3/3 |

**Reading the table.** Effective prefill includes queueing and any mixed
decode work until the last request receives its first token. Total decode is
the sum of per-request token rates inside the common interval where all
streams are emitting. Decode/request is the median rate in that same interval;
its product with concurrency need not equal the total. TTFT is the median time
to first token. Peak overlap is the observed number of simultaneously active
clients, not the requested concurrency.

**Sustained 8,192-token output check.** Four representative cells were rerun
with 8,192 forced output tokens. Long outputs raise the steady-state aggregate
at C4 and C8 relative to the 2,048-token sweep; C1 stays within noise. The
2,048 matrix is therefore a conservative short-window view, not an inflated
one.

| Input tokens/request | C | Total decode tok/s | Decode/request tok/s | TTFT s | Shared decode s | Success |
|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 37.1 | 37.1 | 0.6 | 220.5 | 1/1 |
| 512 | 4 | 88.0 | 22.0 | 1.6 | 371.9 | 4/4 |
| 512 | 8 | 146.3 | 18.3 | 2.3 | 447.7 | 8/8 |
| 400,000 | 1 | 34.4 | 34.4 | 181.7 | 237.9 | 1/1 |

**Repeatability check.** The 512-token row was rerun in a separate directory
with the same launch and prompts.

| C | First run total decode | Repeat total decode | Delta |
|---:|---:|---:|---:|
| 1 | 36.3 | 35.9 | −1.1% |
| 2 | 62.0 | 60.4 | −2.5% |
| 4 | 75.6 | 80.8 | +6.8% |
| 6 | 79.5 | 76.3 | −4.0% |
| 8 | 116.2 | 116.7 | +0.5% |

C1/C2/C6/C8 repeat inside the protocol's 5% noise band. C4 is the widest
spread (+6.8%), so its aggregate should be read with a wider band than the
other cells.

**Near-limit check.** A single **496,000-token** input with 1,024 forced output
tokens completed in the 500k window: TTFT **230.7 s** (2,149 tok/s effective
prefill), total decode **34.5 tok/s**, shared decode window 29.6 s. This is a
usable-window receipt, not a quality score.

## 5. Experiment history

### 5.1 The A/B that selected FP4/vLLM

The same pair was measured against the TensorFold EXL3 incumbent on
2026-10-17. Both candidates are quantizations of the same base model. The
comparison is recorded in [`EVAL-REPORT.md`](EVAL-REPORT.md); the key result is
that the FP4/vLLM stack needed tuning to become competitive on decode while it
was already faster on prefill.

| Metric (512-token streamed chat) | TensorFold EXL3 | FP4/vLLM stock | FP4/vLLM tuned |
| --- | ---: | ---: | ---: |
| Decode, 1 stream | 61.8 tok/s | 23.7 tok/s | **30.0 tok/s** |
| Decode, 4 streams | 103.2 tok/s | ~45 tok/s | **62.1 tok/s** |
| Decode, 8 streams | — | — | **83.7 tok/s** |
| 195k needle prefill | 1,889 tok/s | 2,193 tok/s | **2,300 tok/s** |

The chat numbers above are from the A/B note and are not directly comparable
to the forced-output matrix in §4. The needle prefill is the cleanest
cross-stack measurement: the FP4 stack is about 22% faster on the same prompt.

### 5.2 Tuning sweep for the FP4 stack

All rows used 512-token streamed completions with thinking low, measured from
`usage.completion_tokens` between the first and last streamed chunk. “4/8
streams” are aggregate tok/s.

| Config | 1 stream | 4 streams | 8 streams |
| --- | ---: | ---: | ---: |
| Stock recipe (MTP 5, seqs 4, batched 4096) | 23.7 | ~45 | — |
| + probabilistic draft sampling (MTP 5) | 23.7 | ~45 | — |
| MTP 3 + seqs 8 | 28.2 | 55.3 | 74.0 |
| + decode-context-parallel 2 (+ interleave 4) | 26.0 | 52.4 | 71.4 (worse, reverted) |
| MTP 2 + batched 8192 + seqs 8 | 30.5 | 60.1 | 79.0 |
| **MTP 1 + batched 8192 + seqs 8 (chosen)** | **30.0** | **62.1** | **83.7** |

Stock MTP accepted about 2.1/5 tokens per step with ~21% draft acceptance.
MTP 2 raised acceptance to ~43%; MTP 1 shifted the balance further toward step
rate. Probabilistic drafting was neutral here. Decode-context parallelism did
not help on this pair. The tuned config is +27% at one stream and +38% at four
over stock in that chat harness.

### 5.3 Operational history

| Experiment | Result | Decision / limit |
| --- | --- | --- |
| Stock recipe on the pair | Loaded and served; decode behind TensorFold | Tune scheduling and MTP. |
| MTP 1 + 8 slots + 8192 batch | Best measured chat decode and the published profile | Keep. |
| DCP 2 | Lower aggregate decode | Reverted. |
| 500k served window | Loads and serves; 1.62M-token KV pool | Published limit. Native 1M not enabled. |
| 195k needle | Correct answer, ~2,300 tok/s prefill | Feature acceptance, not quality. |
| Single-image vision | Correct red-square answer | Single-image only. |
| Anthropic surface | Messages and count_tokens pass; two protocol-shape gaps | Partial qualification. |

## 6. Feature and controller acceptance

| Check | Result | Evidence |
| --- | --- | --- |
| Health / models | pass | `/health` 200; `nvidia/GLM-5.3-Flash-FP4-Local` listed |
| Arithmetic | pass | `19 + 23` → `42` |
| JSON schema | pass | `{"city":"Paris","temperature_c":18,"conditions":"Partly cloudy"}` |
| Tool call + result | pass | `get_weather({"city":"Paris"})` → `22°C, sunny` |
| Single-image vision | pass | red square → `Red` |
| Anthropic count_tokens | pass | 14 input tokens |
| Anthropic messages | pass | one-word reply `Hello.` |
| Prompt reuse, 32k tokens | pass | 13.83 s first → 0.49 s second |
| Needle, 190k tokens | pass | 190,012 prompt tokens; passphrase `violet-harbor-7291` found |

The checks above are smoke acceptance on one boot. They do not establish
quality, multi-image reliability, video support, or production readiness.
The model is a reasoning model; small `max_tokens` budgets can be consumed by
the reasoning block even when the final answer is short.

## 7. Reproduce benchmarks and stop

Capture the launch identity first; the matrix hashes `results/launch.json`.
The running process was started with the upstream recipe plus overrides, so its
command line contains upstream defaults followed by the tuned values; vLLM uses
the last value. The clean equivalent is the copied tuned recipe in `recipe/`.

```bash
python3 benchmarks/capture_env.py
```

Run the main matrix (2,048 forced output tokens; skip cells above the KV
capacity guard):

```bash
OUTPUT_TOKENS=2048 \
PREFILL_SIZES=512,2048,8192,32768,65536,131072,200000,400000 \
CONCURRENCIES=1,2,4,6,8 \
python3 benchmarks/matrix.py
```

Recompute the table from the saved token events at any time, refresh the
stable receipts and run the structural audit:

```bash
./benchmarks/finalize.sh results/matrix-<timestamp>
```

Run the feature checks only when the matrix is idle:

```bash
python3 benchmarks/smoke.py
```

Stop the service and preserve the checkpoint, caches and logs:

```bash
./stop.sh
```

The harness writes:

- `results/launch.json` — model revision, tokenizer/config hashes, image
  digest, exact command, cache size, host/GPU state.
- `results/matrix-<timestamp>/<size>-c<n>.json` — raw per-request events.
- `results/matrix-<timestamp>/summary.json` — one row per completed cell.
- `results/matrix-<timestamp>/TABLE.md` — the rendered table.
- `results/feature-acceptance.json` — smoke results and excerpts.
- `results/needle-190k.json` — long-context retrieval receipt.

## 8. Provenance

| Component | Pinned source / attribution |
| --- | --- |
| Model | [nvidia/GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local), revision `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5`. NVFP4 routed experts, MXFP8/FP8 and BF16 elsewhere. |
| Runtime image | `eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595`, vLLM `0.1.dev21830+g8244d1c44`. |
| Launcher | [eugr/spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) at `0bdc9c9`, recipe copied into `recipe/`. |
| B12X | B12X attention/MoE/linear/loader backends in the image above. |
| Incumbent comparison | [MiaAI-Lab TensorFold recipe](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) at `3fd60dfc`, recorded in `EVAL-REPORT.md`. |
| Hardware | 2 × NVIDIA DGX Spark GB10, 128 GB unified memory each, ConnectX-7 RoCE. |

Weights are downloaded at runtime and retain their upstream license. They are
not included in this repository.

## Study protocol and limitations

The fixed workload, ranking rules, bounded experiment list and evidence audit
are in [`STUDY-PROTOCOL.md`](STUDY-PROTOCOL.md). The short version:

- A throughput winner requires repeatable matched measurements; the first
  matrix is a baseline, not a final sweep.
- Differences under 5% between repeated key cells are treated as noise.
- Forced-output tokens are not quality scores.
- Missing measurements remain unknown; no synthetic number is substituted.
- The 500k window, single-image vision and Anthropic shape are qualified only
  to the extent shown above. Native 1M context, multi-image, video, a
  one-hour soak and rollback remain open.
