# Measured inference matrix

Effective prefill includes queueing and mixed decode work until the last request reaches its first token. Total decode is the sum of per-request token rates inside the common decode interval. Decode/request is the median rate in that same interval. TTFT is the median time to first token.

Synthetic repeated notes with unique prefixes; 8,192-token forced output budget; prefix reuse excluded by a nonce prefix. This is not a quality test.

| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | Decode/request tok/s | TTFT s | Shared decode s | Peak overlap | Success |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 1 | 868.7 | 37.1 | 37.1 | 0.6 | 220.5 | 1 | 1/1 |
| 512 | 4 | 1291.9 | 88.0 | 22.0 | 1.6 | 371.9 | 4 | 4/4 |
| 512 | 8 | 1764.8 | 146.3 | 18.3 | 2.3 | 447.7 | 8 | 8/8 |
| 400,000 | 1 | 2201.4 | 34.4 | 34.4 | 181.7 | 237.9 | 1 | 1/1 |
