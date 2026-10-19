#!/usr/bin/env python3
"""Capture the identity and launch configuration of the running recipe.

Writes `results/launch.json` (hashed by the matrix) and
`results/local-fp4-mtp1-config.json` (the short published configuration).
No `.env` file is read and no environment values are printed.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
CONTAINER = os.environ.get("CONTAINER", "vllm_node")
WORKER = os.environ.get("WORKER_HOST", "169.254.108.141")


def run(command: str, timeout: int = 30) -> str:
    try:
        return subprocess.run(
            command, shell=True, check=True, text=True, capture_output=True,
            timeout=timeout,
        ).stdout.strip()
    except Exception as error:  # noqa: BLE001 - capture, do not fail the report
        return f"<unavailable: {error}>"


def get_json(path: str) -> dict:
    with urllib.request.urlopen(API_BASE + path, timeout=20) as response:
        return json.loads(response.read())


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


snapshot = Path(sorted(glob.glob(
    str(Path.home() / ".cache/huggingface/hub/"
        "models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/*")))[-1])
model = get_json("/v1/models")["data"][0]
version = get_json("/version").get("version")
metrics = urllib.request.urlopen(API_BASE + "/metrics", timeout=20).read().decode()
cache_line = next(
    (line for line in metrics.splitlines() if line.startswith("vllm:cache_config_info{")),
    "",
)
cache_info = dict(re.findall(r'(\w+)="([^"]*)"', cache_line))
processes = run(f"docker exec {CONTAINER} ps -eo pid,args")
command = next(
    (line.split(None, 1)[1] for line in processes.splitlines() if "vllm serve" in line),
    "<not found>",
)
image_id = run(f"docker inspect {CONTAINER} --format '{{{{.Image}}}}'")
image_digest = run(
    f"docker image inspect {image_id} --format '{{{{index .RepoDigests 0}}}}'"
)

local_gpu = run(
    "nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.max.sm,"
    "power.draw,power.limit,temperature.gpu --format=csv,noheader"
)
worker_gpu = run(
    f"ssh -o BatchMode=yes -o ConnectTimeout=5 {WORKER} "
    "'hostname; nvidia-smi --query-gpu=name,memory.total,clocks.sm,clocks.max.sm,"
    "power.draw,power.limit,temperature.gpu --format=csv,noheader'",
    timeout=20,
)

launch = {
    "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    "recipe": "GLM-5.3-Flash-FP4-Local on 2x DGX Spark (GB10), TP2, B12X",
    "api_model": model["id"],
    "max_model_len": model.get("max_model_len"),
    "vllm_version": version,
    "image_id": image_id,
    "image_digest": image_digest,
    "model_revision": snapshot.name if snapshot.exists() else None,
    "model_snapshot": str(snapshot),
    "tokenizer_sha256": sha256(snapshot / "tokenizer.json"),
    "model_config_sha256": sha256(snapshot / "config.json"),
    "command": command,
    "cache_config": cache_info,
    "kv_cache_size_tokens": (
        int(float(cache_info["kv_cache_size_tokens"]))
        if cache_info.get("kv_cache_size_tokens") else None
    ),
    "host": {
        "hostname": run("hostname"),
        "kernel": run("uname -a"),
        "head_gpu": local_gpu,
        "worker": worker_gpu,
    },
}
(RESULTS / "launch.json").write_text(json.dumps(launch, indent=2) + "\n")

def value(flag: str, default: str = "—") -> str:
    matches = re.findall(re.escape(flag) + r"\s+(\S+)", command)
    return matches[-1] if matches else default

specs = re.findall(r"--speculative-config\s+(\{.*?\})\s+--", command + " --")
config = {
    "model": model["id"],
    "model_revision": launch["model_revision"],
    "runtime_image": image_digest,
    "vllm_version": version,
    "hardware": "2x NVIDIA DGX Spark (GB10), 128 GB unified memory each, CX7 RoCE link",
    "tensor_parallel": value("--tensor-parallel-size"),
    "decode_context_parallel": value("--decode-context-parallel-size"),
    "max_model_len": value("--max-model-len"),
    "max_num_seqs": value("--max-num-seqs"),
    "max_num_batched_tokens": value("--max-num-batched-tokens"),
    "gpu_memory_utilization": value("--gpu-memory-utilization"),
    "block_size": value("--block-size"),
    "kv_cache_dtype": value("--kv-cache-dtype"),
    "attention_backend": value("--attention-backend"),
    "moe_backend": value("--moe-backend"),
    "load_format": value("--load-format"),
    "speculative_config": specs[-1] if specs else None,
    "kv_cache_size_tokens": launch["kv_cache_size_tokens"],
    "output_tokens": int(os.environ.get("OUTPUT_TOKENS", "2048")),
    "ignore_eos": True,
    "prefix_caching": "--enable-prefix-caching" in command,
    "chunked_prefill": "--enable-chunked-prefill" in command,
}
(RESULTS / "local-fp4-mtp1-config.json").write_text(json.dumps(config, indent=2) + "\n")
print(json.dumps(config, indent=2))
