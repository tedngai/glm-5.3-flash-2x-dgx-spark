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
| KV capacity | **1,617,857 tokens** at the measured start (453 blocks). Three 500k requests fit; eight do not. |
| Admission slots | `max-num-seqs 8`. The matrix records actual overlap; a requested C8 is not counted as eight streams without it. |
| Speculative decoding | MTP with **one** draft token, `moe_backend=humming`, `attention_backend=B12X`. |
| Speed | See the measured matrix in §4. Total shared-window decode scales from one to eight streams up to the tested 131k-input cells. |
| Long context | 190k needle retrieval passed; 400k inputs are in the matrix; 496k near-limit is a separate receipt. |
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
At 200k inputs the measured recipe admits six; at 400k it admits three. The
matrix skips cells above 1.55M aggregate prompt tokens rather than allowing
the scheduler to queue or retract them.

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

<!-- MATRIX_SUMMARY -->

**Reading the table.** Effective prefill includes queueing and any mixed
decode work until the last request receives its first token. Total decode is
the sum of per-request token rates inside the common interval where all
streams are emitting. Decode/request is the median rate in that same interval;
its product with concurrency need not equal the total. TTFT is the median time
to first token. Peak overlap is the observed number of simultaneously active
clients, not the requested concurrency.

<!-- SUSTAINED_SUMMARY -->

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

<!-- FEATURE_TABLE -->

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

Recompute the table from the saved token events at any time:

```bash
python3 benchmarks/common_window.py results/matrix-<timestamp>
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
- `results/needle-195k.json` — long-context retrieval receipt.

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
