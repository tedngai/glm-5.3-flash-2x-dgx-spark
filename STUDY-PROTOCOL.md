# Study protocol

This study applies a fixed, auditable measurement protocol to GLM-5.3-Flash on
two NVIDIA DGX Sparks. It is a reference for one tested environment, not a
global SOTA claim. Performance and quality are separate milestones: a fast
forced-output sweep is not evidence of answer quality.

## Sources and scope

| Source | Pinned revision | Application |
|---|---|---|
| [NVIDIA GLM-5.3-Flash-FP4-Local](https://huggingface.co/nvidia/GLM-5.3-Flash-FP4-Local) | `76348ddb2be2e35e39ecfc00e75dc07703f4f3d5` | Pre-quantized checkpoint: NVFP4 routed experts, MXFP8/FP8 and BF16 elsewhere. No local requantization. |
| [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) | `0bdc9c9` | Cluster launcher and B12X-serving recipe. The tuned recipe is copied into `recipe/` so the published settings do not depend on a moving upstream default. |
| [B12X vLLM image](https://hub.docker.com/r/eugr/spark-vllm-b12x) | `eugr/spark-vllm-b12x@sha256:03a956544e9f9b4b7ba91690469fbe7848bdbef29a7d2e0f5788368467bb3595` | vLLM `0.1.dev21830+g8244d1c44` with B12X attention, MoE, linear and loader backends. |
| [TensorFold comparison](https://github.com/MiaAI-Lab/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold) | `3fd60dfc` | Incumbent EXL3 engine on the same pair. Used only for the A/B history in `EVAL-REPORT.md`; no numbers from it are merged into the FP4 matrix. |

## Fixed experiment contract

- Checkpoint `nvidia/GLM-5.3-Flash-FP4-Local` revision `76348ddb`, served as
  `nvidia/GLM-5.3-Flash-FP4-Local`.
- Two DGX Sparks (GB10, 128 GB unified memory each) over the ConnectX-7 RoCE
  link, TP2, decode-context-parallel 1, FP8 KV cache, MTP speculative decoding
  with one draft token.
- Context 500,000 tokens, `max-num-seqs 8`, `max-num-batched-tokens 8192`,
  GPU memory utilization 0.87, prefix caching and chunked prefill enabled.
- Main matrix: 512 / 2,048 / 8,192 / 32,768 / 65,536 / 131,072 / 200,000 /
  400,000 input tokens at the highest concurrency that fits the measured KV
  pool, with 2,048 forced output tokens per request. Cells above 1.55M
  aggregate prompt tokens are skipped rather than allowed to queue or retract.
- Sustained check: 8,192 forced output tokens at representative cells. The
  129-641 token window is used for per-request decode rate so startup and
  prefill do not dominate the number.
- Every prompt is exact-length token IDs assembled from a unique nonce, a
  repeated reference-text filler and a fixed instruction suffix. The nonce
  prevents prefix-cache reuse from turning a cell into a cache-hit test.
- Raw streamed token counts, timings, finish reasons and reported prompt sizes
  are saved per request. Missing measurements stay unknown; no synthetic
  number is substituted.

## Metrics

- **Effective prefill tok/s**: total input tokens divided by the time until the
  last request in the wave receives its first token. It includes queueing and
  any prefill/decode overlap.
- **Total decode tok/s**: sum of per-request token rates inside the common
  interval where every request in the wave is emitting tokens.
- **Decode/request tok/s**: median per-request rate in that same common
  interval. Its product with concurrency need not equal the total.
- **TTFT**: median wall-clock time from request start to the first generated
  token.
- **Shared decode seconds**: length of the common decode interval.

A cell is a success only if every request completes with the requested token
count and no retraction. Peak client overlap is recorded so a requested C8 is
not mistaken for eight simultaneously active streams.

## Bounded experiments before widening the sweep

| Order | Experiment | Gate |
|---|---|---|
| 1 | Reproduce the tuned profile | All matrix cells complete; no errors, no retractions, actual overlap recorded. |
| 2 | Repeat key cells | 512/C1, 512/C8, 8,192/C8 and 400,000/C1 repeated; differences under 5% are treated as noise. |
| 3 | Stock vs tuned | Same boot and prompts for MTP 5 / seqs 4 / batched 4096 vs the published tuned profile. |
| 4 | Thinking-off template | The shipped chat template always opens a thinking block; a modified template is required for a fair chat-decode comparison. |
| 5 | Context ceiling | Raise `max-model-len` toward the model's native 1,048,576 only after long-context memory and quality checks. The current recipe stops at 500,000. |
| 6 | Vision | Single-image path is qualified; multi-image ordering, video and tool-result media remain open. |
| 7 | Soak and rollback | One-hour representative traffic, then a documented restart/rollback exercise. |

At most ten new configurations in a tuning round. A failed cell ends that
branch, not the study. Two changes without a repeatable gain trigger profiling
rather than a wider blind sweep.

## Ranking and quality

Rank only completed candidates. Total shared-window decode is the primary
throughput number; prefill, TTFT, per-request decode and capacity stay visible
as trade-offs. Reject a regression above 5% in any tested bucket unless it is
explicitly presented as a trade-off and repeated.

Forced-output prompts are performance tests, not quality scores. Quality checks
use natural termination and fixed prompts. Tool calls, JSON schema adherence,
vision, needle retrieval and the Anthropic surface are smoke acceptance, not a
full quality evaluation.

## Evidence audit

The matrix writer stores:

- `results/launch.json`: model revision, tokenizer/config hashes, image digest,
  exact launch command, cache size, host and GPU state.
- `results/matrix-<timestamp>/<size>-c<n>.json`: raw per-request token events.
- `results/matrix-<timestamp>/summary.json`: one summary per completed cell.
- `results/matrix-<timestamp>/TABLE.md`: the rendered table.
- `results/feature-acceptance.json`: smoke results and excerpts.
- `results/local-fp4-mtp1-config.json`: the short published configuration.

`benchmarks/common_window.py` recomputes every summary from the raw events.
The audit does not upgrade incomplete cells to production acceptance.
