# Results

Published receipts for the tuned GLM-5.3-Flash FP4 recipe. The benchmark code
that produced them is in `../benchmarks/`.

| File | Contents |
|---|---|
| `launch.json` | Model revision, tokenizer/config hashes, image digest, exact running command, cache size, host/GPU state. The matrix hashes this file. |
| `local-fp4-mtp1-config.json` | Short published configuration extracted from the launch. |
| `local-fp4-mtp1-summary.json` | One row per completed 2,048-token matrix cell. |
| `local-fp4-mtp1-table.md` | Rendered matrix table (same cells as the README). |
| `matrix-<timestamp>/` | Raw per-request token events, per-cell summaries and the rendered table. |
| `sustained-8192/` | 8,192-token sustained cells at 512/C1, 512/C4, 512/C8 and 400k/C1. |
| `near-limit-496k/` | Single-request near-limit case in the 500k served window. |
| `repeat-512/` | Repeat of the 512 row for a noise band. |
| `feature-acceptance.json` | Arithmetic, JSON schema, tool call/result, single-image vision, Anthropic Messages, prompt reuse and needle checks. |
| `needle-190k.json` | Long-context retrieval receipt. |
| `protocol-audit.json` | Structural audit: exact prompt/output counts, no failed requests, matching manifest hashes. |

All throughput cells use exact-length token-ID prompts, unique nonce prefixes
(no prefix-cache reuse), forced output and greedy sampling. They are
performance tests, not quality scores.
