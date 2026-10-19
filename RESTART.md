# Restart / switch guide — GLM-5.3-Flash on 2× DGX Spark

> The published wrapper is `./start.sh` / `./stop.sh` in this directory, using
> `recipe/glm-5.3-flash-tuned.yaml`. The commands below show the underlying
> launcher steps and the TensorFold switch.

Date: 2026-10-17 · Head `promaxgb10-42ce` (169.254.61.168) · Worker `promaxgb10-42f3` (169.254.108.141)
Companion to `EVAL-REPORT.md`. Both stacks serve on port **8000** (host network).

**Only one stack can run at a time** — each wants ~116 GiB of the 128 GB unified
memory per Spark. **Nothing auto-starts after a reboot** (verified 2026-10-17:
docker restart policy `no`, `AutoRemove=true`, no systemd unit, no cron; docker
itself is enabled at boot). Both checkpoints and the `vllm-node-b12x` image are
on disk, so restarts never re-download.

Paths: TensorFold `~/data/glm53-flash-tensorfold` · FP4/vLLM `~/data/spark-vllm-docker`
Link: `enp1s0f1np1`, RoCE `rocep1s0f1,rocepP2p1s0f1` (both active).

## 0. Preflight (do this first, always)

```bash
# on the head
ip -4 addr show enp1s0f1np1 | grep inet          # expect 169.254.61.168/16
ssh -o BatchMode=yes tngai@169.254.108.141 hostname   # -> promaxgb10-42f3
free -g | head -2                                 # expect ~115+ GiB available
df -h / | tail -1                                 # ~250 GB free (head), ~350 GB (worker)
docker ps                                         # expect no vllm_node / glm53-flash-tf
```

Link-local IPs can change after a reboot or cable move. If they did, refresh the
saved cluster config (accept the prompts with Enter):

```bash
cd ~/data/spark-vllm-docker
./run-recipe.sh --discover -n <head-ip>,<worker-ip>   # e.g. 169.254.61.168,169.254.108.141
```

## 1. FP4 vLLM — tuned config (A/B winner)

Model `nvidia/GLM-5.3-Flash-FP4-Local` · image `vllm-node-b12x`
(= `eugr/spark-vllm-b12x:latest`) · API model id = `nvidia/GLM-5.3-Flash-FP4-Local`.

Start (foreground; use tmux or the nohup form):

```bash
cd ~/data/spark-vllm-docker
./run-recipe.sh recipes/glm-5.3-flash.yaml -- \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"humming","attention_backend":"B12X"}'
```

Detached variant:

```bash
cd ~/data/spark-vllm-docker
nohup ./run-recipe.sh recipes/glm-5.3-flash.yaml -- \
  --max-num-seqs 8 --max-num-batched-tokens 8192 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1,"moe_backend":"humming","attention_backend":"B12X"}' \
  > ~/data/glm53-flash-fp4/launch-$(date +%Y%m%d-%H%M).log 2>&1 &
```

- Startup: **~6–8 min** (model load + CUDA graphs). Wait for readiness:

```bash
for i in $(seq 1 60); do curl -sf http://127.0.0.1:8000/v1/models >/dev/null && echo ready && break; sleep 10; done
curl -s http://127.0.0.1:8000/v1/models          # check
curl -s http://127.0.0.1:8000/health
```

- Stop: `cd ~/data/spark-vllm-docker && ./launch-cluster.sh stop`
- Logs: `docker logs -f vllm_node` (head) · `ssh tngai@169.254.108.141 docker logs -f vllm_node` (worker)
- Tuned numbers: 30.0 / 62.1 / 83.7 tok/s decode at 1 / 4 / 8 streams; ~2,300 tok/s prefill (see `EVAL-REPORT.md`).
- Recipe defaults are stock (MTP 5, seqs 4, batched 4096); the flags above are the
  tuning result — keep them if you want the measured numbers.
- Optional: add `--enable-prompt-tokens-details` so `usage.cached_tokens` is reported.
- First run on a fresh machine / missing artifacts: append `--setup` once.

## 2. TensorFold EXL3 — incumbent

Stack `~/data/glm53-flash-tensorfold` (TensorFold v0.6.0 + `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold`
+ DFlash2 drafter) · API model id = `glm-5.3-local`.

**Stop the FP4 cluster first** (section 1), then:

```bash
cd ~/data/glm53-flash-tensorfold
./start.sh              # start both ranks; first run compiles kernels ~6–7 min, later ~2 min
./stop.sh               # stop both ranks (saves each rank's log, gzipped)
./start.sh restart      # apply changed settings/env
```

- Verify: `curl -s http://127.0.0.1:8000/health` → `{"ok": true, "backend": "tensorfold", ...}`;
  `curl -s http://127.0.0.1:8000/v1/models`.
- Logs: `docker logs -f glm53-flash-tf` (head) · `ssh tngai@169.254.108.141 docker logs -f glm53-flash-tf` (worker);
  gzipped history in `~/.cache/tensorfold-glm53/logs/` on each node.
- Known issue #163 (rare split-prefill wedge): `./start.sh restart`; if it recurs, `SPLIT=0 ./start.sh restart`.
- Memory: ~116 GiB free per Spark required. Do not start while `vllm_node` is up.

## 3. Switching checklist

1. Stop the running stack (§1 `launch-cluster.sh stop` or §2 `./stop.sh`).
2. Confirm both nodes are clear: `docker ps` shows nothing, `free -g` shows ~115+ GiB available on each Spark.
3. Start the other stack (§1 or §2) and wait for readiness before sending traffic.

Both bind port 8000 — if you see "address already in use", a container from the
other stack is still running; stop it first.

## 4. After a reboot

- Nothing starts automatically. Docker itself does (enabled).
- Re-run **§0 preflight** (the CX7 IPs may have changed), then **§1** or **§2**.
- Containers are `AutoRemove=true`, so nothing is left half-stopped — just relaunch.

## 5. Troubleshooting quick hits

| Symptom | Fix |
|---|---|
| port 8000 in use | another stack's container is still up: `docker ps`, stop it |
| worker unreachable | check cable/`ip -4 addr`; if IP changed, re-run discovery (§0) |
| load stalls / host OOM-ish | `free -g`; ensure the other stack is stopped; if swap is exhausted, reboot both Sparks and retry |
| HF auth warning / slow download | stored token is stale: `hf auth login --force` (a valid token lives in the project `.env` files) |
| model not found on start | `run-recipe.sh ... --setup` once to verify/download |
