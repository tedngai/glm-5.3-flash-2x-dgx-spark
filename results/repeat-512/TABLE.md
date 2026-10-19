# Measured inference matrix

Effective prefill includes queueing and mixed decode work until the last request reaches its first token. Total decode is the sum of per-request token rates inside the common decode interval. Decode/request is the median rate in that same interval. TTFT is the median time to first token.

Synthetic repeated notes with unique prefixes; 2,048-token forced output budget; prefix reuse excluded by a nonce prefix. This is not a quality test.

| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | Decode/request tok/s | TTFT s | Shared decode s | Peak overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 843.7 | 35.9 | 35.9 | 0.6 | 56.9 | 1 | 1/1 |
| 512 | 2 | 1357.4 | 60.4 | 30.2 | 0.8 | 67.7 | 2 | 2/2 |
| 512 | 4 | 1282.1 | 80.8 | 20.2 | 1.5 | 101.2 | 4 | 4/4 |
| 512 | 6 | 1506.4 | 76.3 | 12.8 | 2.0 | 159.1 | 6 | 6/6 |
| 512 | 8 | 1548.0 | 116.7 | 14.6 | 2.6 | 139.9 | 8 | 8/8 |
