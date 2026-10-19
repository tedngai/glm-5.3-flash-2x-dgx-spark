# Benchmarks

All scripts are Python 3.10+ and use only the standard library plus
`tokenizers` (see `requirements.txt`). They talk to the running vLLM server over
its OpenAI-compatible API.

| Script | Purpose |
|---|---|
| `capture_env.py` | Writes `results/launch.json` and the short config receipt. The matrix hashes `launch.json` so every cell is tied to one launch identity. |
| `matrix.py` | Sends exact-length token-ID prompts at the requested input sizes and concurrency, saves raw streamed token events, and writes `summary.json` + `TABLE.md`. |
| `common_window.py` | Recomputes summaries from saved raw events; use it to audit a published matrix. |
| `smoke.py` | Bounded feature checks: arithmetic, JSON schema, tool call + result, single-image vision, Anthropic Messages, prompt reuse and a 190k needle. |

## Matrix environment

| Variable | Default | Meaning |
|---|---|---|
| `API_BASE` | `http://127.0.0.1:8000` | vLLM OpenAI endpoint. |
| `MODEL` | `nvidia/GLM-5.3-Flash-FP4-Local` | Served model ID. |
| `TOKENIZER_PATH` | auto-detect | `tokenizer.json` used to build exact-length prompts. |
| `PREFILL_SIZES` | `512,2048,8192,32768,65536,131072,200000,400000` | Input lengths. |
| `CONCURRENCIES` | `1,2,4,6,8` | Requested burst sizes. |
| `OUTPUT_TOKENS` | `2048` | Forced output tokens per request. |
| `IGNORE_EOS` | `1` | Generate exactly the budget. |
| `MAX_PROMPT_TOKENS` | `496000` | Per-request cap; leaves room in a 500k window. |
| `MAX_AGGREGATE_PROMPT_TOKENS` | `1550000` | Skips cells that would exceed KV headroom. |
| `CASE_TIMEOUT_S` | `3600` | Per-request socket timeout. |
| `RESULTS_DIR` | `results/matrix-<timestamp>` | Output directory. |
| `CONTINUE_ON_ERROR` | unset | Set to `1` to keep going after a failed wave. |

## Why exact token IDs

Sending text and trusting the tokenizer count makes prompt-length comparisons
soft. `matrix.py` builds a prompt from a unique nonce, a repeated reference
filler and a fixed instruction, trims it to the requested number of token IDs,
and verifies the server-reported `prompt_tokens`. The nonce also prevents the
prefix cache from turning a cell into a cache-hit measurement.

## Raw event shape

Each request record contains cumulative `{seconds, count}` events from
`stream_options.continuous_usage_stats`, plus `first_token_s`, `finished_s`,
`ttft_s`, `decode_window_tps`, `output_tokens` and any error. The summary is
derived only from those events; `common_window.py` can reproduce it.
