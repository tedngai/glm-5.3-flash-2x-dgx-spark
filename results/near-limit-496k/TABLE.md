# Measured inference matrix

Effective prefill includes queueing and mixed decode work until the last request reaches its first token. Total decode is the sum of per-request token rates inside the common decode interval. Decode/request is the median rate in that same interval. TTFT is the median time to first token.

Synthetic repeated notes with unique prefixes; 1,024-token forced output budget; prefix reuse excluded by a nonce prefix. This is not a quality test.

| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | Decode/request tok/s | TTFT s | Shared decode s | Peak overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 496,000 | 1 | 2149.5 | 34.5 | 34.5 | 230.7 | 29.6 | 1 | 1/1 |
