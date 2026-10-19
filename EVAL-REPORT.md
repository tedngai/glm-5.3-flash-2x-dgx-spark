# GLM-5.3-Flash on 2× DGX Spark — A/B evaluation notes (historical)

> This is the 2026-10-17 A/B note that selected the FP4/vLLM stack. The
> published recipe, current measurements and caveats are in
> [`README.md`](README.md); the fixed protocol is in
> [`STUDY-PROTOCOL.md`](STUDY-PROTOCOL.md). The tuned launch is now the copied
> recipe at `recipe/glm-5.3-flash-tuned.yaml`.

Date: 2026-10-17 · Pair: promaxgb10-42ce (head) + promaxgb10-42f3 (worker), CX7 link

Both candidates are quantizations of the same base model: ZAI GLM-5.3-Flash
(320B total / 18B active MoE, hybrid sparse+linear attention, mHC, 1M ctx, multimodal).

| | A — TensorFold EXL3 (incumbent) | B — NVIDIA FP4 (new) |
|---|---|---|
| Weights | Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold rev `078455ff`, 164 GB | nvidia/GLM-5.3-Flash-FP4-Local rev `76348ddb`, 166 GB |
| Engine | TensorFold v0.6.0 (107-patch image) + DFlash2 draft | vLLM b12x (`eugr/spark-vllm-b12x:latest`, build `v0.1.dev21830+g8244d1c44`) |
| Served cfg | ctx 1,048,576 · parallel 4 · fp8 KV | tuned: ctx 500,000 · max-num-seqs 8 · fp8 KV · MTP x1 |
| KV capacity | ~2.58M-token pool (observed) | 1,776,119 tokens (3.55× @500K) |
| API name | `glm-5.3-local` | `nvidia/GLM-5.3-Flash-FP4-Local` |

## Check battery (A = 2026-10-16 baseline from tools-checks-20261016.log)

| Check | A (TensorFold) | B (FP4/vLLM) | Verdict |
|---|---|---|---|
| toolcheck (array tool arg) | OK | OK (also on tuned config) | parity |
| needle ~195k tok | CORRECT, prefill 103.2 s ≈ 1,889 tok/s | CORRECT, prefill 88.9 s ≈ 2,193 tok/s (stock); **86.7 s ≈ 2,300 tok/s (tuned)** | parity, **B +22% prefill** |
| prompt reuse (33k agent conv) | OK, 99% cached reported | 99.3% hit rate (metrics); tool prints FAIL — vLLM needs `--enable-prompt-tokens-details` to report `cached_tokens` | parity in behaviour |
| Anthropic surface | all ok (6/6) | 4/6: message, count_tokens, streaming tool_use, tool_result, stop_sequences OK; **400 error envelope** and **chunked request body** differ | partial |
| end_of_turn (FR coding, thinking off) | exit 0: 4/48 cut; P(EOT) mean 0.68, min 0.23 | 13/48 cut; P(EOT) 0.00 — **but ran with thinking on** (see caveat 1) | not comparable as-run |
| decode, 512-tok streamed | 61.8 tok/s (1) / 103.2 (4) [prose] | **30.0 / 62.1 (4) / 83.7 (8) aggregate** after tuning | **B ≈ 1.7–2.1× slower** |

Anthropic details for B:
- 400 shape: `{"error":{"message":...,"type":"Bad Request",...}}` vs Anthropic
  `{"type":"error","error":{"type":"invalid_request_error",...}}`.
- `Transfer-Encoding: chunked` request body → HTTP 200 with empty body.

## Decode tuning sweep (B), 512-token streamed completions, thinking low

Per-stream decode in tok/s measured from usage.completion_tokens between first and
last streamed chunk; "4/8 streams" columns are aggregate tok/s.

| Config | 1 stream | 4 streams | 8 streams |
|---|---|---|---|
| stock recipe (MTP 5, seqs 4, batched 4096) | 23.7 | ~45 | — |
| + probabilistic draft sampling (MTP 5) | 23.7 | ~45 | — |
| MTP 3 + seqs 8 | 28.2 | 55.3 | 74.0 |
| + decode-context-parallel 2 (+ interleave 4) | 26.0 | 52.4 | 71.4 (worse — reverted) |
| MTP 2 + batched 8192 + seqs 8 | 30.5 | 60.1 | 79.0 |
| **MTP 1 + batched 8192 + seqs 8 (chosen)** | **30.0** | **62.1** | **83.7** |

Notes: stock MTP accepted ~2.1/5 tokens per step with ~21% draft acceptance;
MTP 2 raised draft acceptance to ~43% and MTP 1 shifted the balance further toward
step rate. Probabilistic drafting was neutral here. DCP did not help on this pair.
The tuned config is **+27% / +38%** over stock at 1 / 4 streams.

## Key caveats

1. **Thinking mode mismatch**: the FP4 model's official chat template always opens
   `<think>` (`reasoning_effort` = low/high/max only; default max). The tools'
   `enable_thinking:false` is a no-op there, so `end_of_turn` (and the decode bench)
   ran with reasoning on. TensorFold's own prompt layer supports thinking-off.
   A fair rerun needs a modified chat template passed to vLLM.
2. Even after tuning, B's decode remains ~1.7–2.1× behind TensorFold's purpose-built
   engine on this pair, while prefill is ~22% faster. The gap is engine maturity,
   not the quantization.
3. Recipe defaults: 500K ctx (model supports more), no usage-details flag.
4. Both checkpoints now live on both nodes (head/worker).

## Tuned launch command (currently running)

```sh
cd ~/data/spark-vllm-docker
./run-recipe.sh recipes/glm-5.3-flash.yaml -- \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"humming","attention_backend":"B12X"}'
```

## Next experiments (if wanted)

- Thinking-off template override → fair rerun of `end_of_turn` + decode bench.
- Add `--enable-prompt-tokens-details` for `cached_tokens` parity.
- Try 16 sequences / larger batches for multi-user throughput (KV pool headroom allows).
- Vision: both stacks include a vision path (untested here).

## Operations cheat-sheet

Full copy-paste restart/switch guide (both stacks, post-reboot checks): **`RESTART.md`** in this directory.

- B stop / start: `cd ~/data/spark-vllm-docker && ./launch-cluster.sh stop` / command above
- A restore: `cd ~/data/glm53-flash-tensorfold && ./start.sh` (after stopping B)
- B logs: `docker logs -f vllm_node` (head and worker); this dir's `launch-run*.log` too.
- Head/worker free disk after staging: 253 GB / 352 GB.

## Artifacts in this directory

- `check-toolcheck.log`, `check-toolcheck-tuned.log`, `check-needle.log`,
  `check-needle-uncached.log`, `check-needle-tuned.log`, `check-prompt_reuse*.log`,
  `check-anthropiccheck.log`, `check-end_of_turn.log`
- `check-decode-*.log` (stock, probabilistic, spec3+seq8, DCP2, spec2, spec1)
- `metrics-*.txt` (KV/prefix-cache/prefill/spec-decode counters)
- `dry-run.log`, `setup-launch.log`, `launch-run*.log`, `model-copy.log`
- `needle-vllm.py`, `end_of_turn-vllm.py`, `bench_decode.py` (minimally patched copies
  for vLLM; originals in `~/data/glm53-flash-tensorfold/tools`)

Note: `~/.cache/huggingface/token` is stale — `hf auth login --force` recommended
(a valid token currently lives in the project `.env` files).
