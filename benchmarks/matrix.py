#!/usr/bin/env python3
"""Measured burst-serving matrix for GLM-5.3-Flash FP4 on 2x DGX Spark.

The harness sends exact-length token-ID prompts to vLLM's OpenAI-compatible
`/v1/completions` endpoint, requests a fixed number of forced output tokens,
and records cumulative token counts from every streamed chunk.  It writes raw
per-request events plus a summary for each input-length / concurrency cell.

This is a throughput and latency test, not a quality evaluation.  Every prompt
starts with a unique nonce so that prefix caching cannot turn a cell into a
cache-hit measurement.

Environment:
  API_BASE          default http://127.0.0.1:8000
  MODEL             default nvidia/GLM-5.3-Flash-FP4-Local
  TOKENIZER_PATH    tokenizer.json; auto-detected from MODEL_PATH/HF cache
  MODEL_PATH        optional local snapshot directory
  RESULTS_DIR       output directory; default results/matrix-<timestamp>
  PREFILL_SIZES     default 512,2048,8192,32768,65536,131072,200000,400000
  CONCURRENCIES     default 1,2,4,6,8 (cells above KV capacity are skipped)
  OUTPUT_TOKENS     default 2048
  IGNORE_EOS        default 1
  MAX_PROMPT_TOKENS default 496000 (leaves room for output in a 500k window)
  MAX_AGGREGATE_PROMPT_TOKENS default 1550000 (KV-pool headroom guard)
  CASE_TIMEOUT_S    default 3600
  SKIP_EXISTING     default 0; reuse completed case files in RESULTS_DIR
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time
import urllib.error
import urllib.request
import uuid

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
API_BASE = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "nvidia/GLM-5.3-Flash-FP4-Local")
OUTPUT_TOKENS = int(os.environ.get("OUTPUT_TOKENS", "2048"))
IGNORE_EOS = os.environ.get("IGNORE_EOS", "1") == "1"
CASE_TIMEOUT = float(os.environ.get("CASE_TIMEOUT_S", "3600"))
SKIP_EXISTING = os.environ.get("SKIP_EXISTING", "0") == "1"
MAX_PROMPT_TOKENS = int(os.environ.get("MAX_PROMPT_TOKENS", "496000"))
MAX_AGGREGATE_PROMPT_TOKENS = int(os.environ.get(
    "MAX_AGGREGATE_PROMPT_TOKENS", "1550000"))
SIZES = [int(x) for x in os.environ.get(
    "PREFILL_SIZES", "512,2048,8192,32768,65536,131072,200000,400000").split(",")]
CONCURRENCIES = [int(x) for x in os.environ.get(
    "CONCURRENCIES", "1,2,4,6,8").split(",")]
STAMP = time.strftime("%Y%m%dT%H%M%S")
FOLDER = Path(os.environ.get("RESULTS_DIR", str(ROOT / "results" / f"matrix-{STAMP}")))
FOLDER.mkdir(parents=True, exist_ok=True)


def find_tokenizer() -> Path:
    explicit = os.environ.get("TOKENIZER_PATH")
    if explicit:
        return Path(explicit)
    model_path = os.environ.get("MODEL_PATH")
    if model_path and (Path(model_path) / "tokenizer.json").exists():
        return Path(model_path) / "tokenizer.json"
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    matches = sorted(hub.glob("models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/*/tokenizer.json"))
    if not matches:
        raise SystemExit("tokenizer.json not found; set TOKENIZER_PATH")
    return matches[-1]


TOKENIZER_FILE = find_tokenizer()
TOK = Tokenizer.from_file(str(TOKENIZER_FILE))
TOKENIZER_SHA256 = hashlib.sha256(TOKENIZER_FILE.read_bytes()).hexdigest()
FILLER = TOK.encode(
    "Reference notes: the cache stores recently accessed entries. "
    "An implementation should maintain ordering, handle replacement and "
    "validate its invariants.\n",
    add_special_tokens=False,
).ids
SUFFIX = TOK.encode(
    "\nNow write a complete Python LRU cache module with a doubly linked list "
    "and dictionary, including get, put, delete, iteration, resize, clear, "
    "invariant validation, detailed docstrings and ten usage examples. "
    "Return code only. Implement all methods fully.\n",
    add_special_tokens=False,
).ids


def prompt_ids(size: int, nonce: str) -> list[int]:
    prefix = TOK.encode(
        f"session {nonce}\nRead these notes before continuing.\n",
        add_special_tokens=False,
    ).ids
    room = size - len(prefix) - len(SUFFIX)
    if room < 0:
        raise ValueError(f"prompt size {size} is smaller than the fixed wrapper")
    filler = (FILLER * (room // len(FILLER) + 1))[:room]
    ids = prefix + filler + SUFFIX
    assert len(ids) == size, (len(ids), size)
    return ids


def stream_one(size: int, index: int, wave_start: float) -> dict:
    nonce = uuid.uuid4().hex
    ids = prompt_ids(size, nonce)
    started = time.monotonic() - wave_start
    record: dict = {
        "index": index,
        "prompt_tokens_requested": size,
        "nonce": nonce,
        "started_s": started,
        "events": [],
        "output_budget": OUTPUT_TOKENS,
        "ignore_eos": IGNORE_EOS,
    }
    body = {
        "model": MODEL,
        "prompt": ids,
        "max_tokens": OUTPUT_TOKENS,
        "temperature": 0,
        "stream": True,
        "stream_options": {
            "include_usage": True,
            "continuous_usage_stats": True,
        },
        "ignore_eos": IGNORE_EOS,
    }
    request = urllib.request.Request(
        API_BASE + "/v1/completions",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    last_count = 0
    saw_done = False
    try:
        with urllib.request.urlopen(request, timeout=CASE_TIMEOUT) as response:
            for raw in response:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                event = json.loads(data)
                usage = event.get("usage") or {}
                count = int(usage.get("completion_tokens") or 0)
                if count > last_count:
                    record["events"].append({
                        "seconds": time.monotonic() - wave_start,
                        "count": count,
                    })
                    last_count = count
                if usage.get("prompt_tokens"):
                    record["prompt_tokens_reported"] = int(usage["prompt_tokens"])
                choices = event.get("choices") or []
                if choices and choices[0].get("finish_reason"):
                    record["finish_reason"] = choices[0]["finish_reason"]
        record["finished_s"] = time.monotonic() - wave_start
        record["saw_done"] = saw_done
        record["output_tokens"] = last_count
        if last_count != OUTPUT_TOKENS:
            record["error"] = (
                f"completion_tokens {last_count} != requested {OUTPUT_TOKENS}"
            )
        if record.get("prompt_tokens_reported") != size:
            record["error"] = (
                "prompt_tokens reported "
                f"{record.get('prompt_tokens_reported')} != requested {size}"
            )
        if not record["events"]:
            record["error"] = "no usage events received"
            return record
        def at(n: int):
            return next((e["seconds"] for e in record["events"] if e["count"] >= n), None)
        record["first_token_s"] = at(1)
        record["ttft_s"] = record["first_token_s"] - started
        low, high = 129, 641
        if OUTPUT_TOKENS >= high:
            left, right = at(low), at(high)
            record["decode_window"] = [low, high]
        else:
            left = at(max(2, OUTPUT_TOKENS // 4))
            right = at(OUTPUT_TOKENS)
            record["decode_window"] = [max(2, OUTPUT_TOKENS // 4), OUTPUT_TOKENS]
        record["decode_window_tps"] = (
            512 / (right - left)
            if OUTPUT_TOKENS >= 641 and left is not None and right is not None and right > left
            else ((record["decode_window"][1] - record["decode_window"][0])
                  / (right - left)
                  if left is not None and right is not None and right > left else None)
        )
        record["overall_decode_tps"] = (
            (last_count - 1) / (record["finished_s"] - record["first_token_s"])
            if last_count > 1 and record["finished_s"] > record["first_token_s"]
            else None
        )
    except urllib.error.HTTPError as error:
        record["error"] = f"HTTP {error.code}: {error.read().decode(errors='replace')[:2000]}"
        record["finished_s"] = time.monotonic() - wave_start
    except Exception as error:  # noqa: BLE001 - benchmark must keep the receipt
        record["error"] = repr(error)
        record["finished_s"] = time.monotonic() - wave_start
    return record


def summarise(size: int, concurrency: int, records: list[dict], manifest: str) -> dict:
    good = [r for r in records if not r.get("error")]
    spans = sorted(
        [(r["first_token_s"], 1) for r in good]
        + [(r["finished_s"], -1) for r in good],
        key=lambda x: (x[0], x[1]),
    )
    active = peak = 0
    for _, delta in spans:
        active += delta
        peak = max(peak, active)
    rates = [r["decode_window_tps"] for r in good if r.get("decode_window_tps")]
    summary = {
        "input_tokens": size,
        "concurrency": concurrency,
        "successes": len(good),
        "peak_client_overlap": peak,
        "manifest_sha256": manifest,
        "tokenizer_sha256": TOKENIZER_SHA256,
        "effective_prefill_tps": (
            size * len(good) / max(r["first_token_s"] for r in good) if good else None
        ),
        "median_ttft_s": (
            statistics.median(r["ttft_s"] for r in good) if good else None
        ),
        "median_decode_tps": statistics.median(rates) if rates else None,
        "errors": [r["error"] for r in records if r.get("error")],
        "total_decode_tps": None,
        "shared_decode_tps_per_request": None,
        "shared_decode_seconds": 0,
    }
    if good and len(good) == concurrency:
        timelines = []
        for record in good:
            previous = 0
            timeline = []
            for event in record["events"]:
                count = event["count"]
                if count > previous:
                    timeline.append((event["seconds"], count - previous))
                previous = count
            timelines.append(timeline)
        if all(timelines):
            begin = max(t[0][0] for t in timelines)
            end = min(t[-1][0] for t in timelines)
            if end > begin:
                shared = [
                    sum(n for timestamp, n in timeline if begin < timestamp <= end)
                    / (end - begin)
                    for timeline in timelines
                ]
                summary.update(
                    total_decode_tps=sum(shared),
                    shared_decode_tps_per_request=statistics.median(shared),
                    shared_decode_seconds=end - begin,
                )
    return summary


def render(summaries: list[dict], output_tokens: int) -> str:
    lines = [
        "# Measured inference matrix",
        "",
        "Effective prefill includes queueing and mixed decode work until the last "
        "request reaches its first token. Total decode is the sum of per-request "
        "token rates inside the common decode interval. Decode/request is the "
        "median rate in that same interval. TTFT is the median time to first token.",
        "",
        f"Synthetic repeated notes with unique prefixes; {output_tokens:,}-token "
        "forced output budget; prefix reuse excluded by a nonce prefix. This is "
        "not a quality test.",
        "",
        "| Input tokens/request | C | Effective prefill tok/s | Total decode tok/s | "
        "Decode/request tok/s | TTFT s | Shared decode s | Peak overlap | Success |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summaries:
        def fmt(key: str) -> str:
            value = item.get(key)
            return f"{value:.1f}" if isinstance(value, (int, float)) else "—"

        lines.append(
            f"| {item['input_tokens']:,} | {item['concurrency']} | "
            f"{fmt('effective_prefill_tps')} | {fmt('total_decode_tps')} | "
            f"{fmt('shared_decode_tps_per_request')} | {fmt('median_ttft_s')} | "
            f"{fmt('shared_decode_seconds')} | {item['peak_client_overlap']} | "
            f"{item['successes']}/{item['concurrency']} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    manifest_file = Path(os.environ.get("MANIFEST_PATH", str(FOLDER.parent / "launch.json")))
    identity = (
        hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        if manifest_file.exists()
        else "missing-launch-manifest"
    )
    print(f"results: {FOLDER}", flush=True)
    print(f"tokenizer: {TOKENIZER_FILE}", flush=True)
    print(f"model: {MODEL}  output={OUTPUT_TOKENS}  ignore_eos={IGNORE_EOS}", flush=True)
    summaries: list[dict] = []
    for size in SIZES:
        if size > MAX_PROMPT_TOKENS:
            print(f"skip {size}: above MAX_PROMPT_TOKENS={MAX_PROMPT_TOKENS}", flush=True)
            continue
        for concurrency in CONCURRENCIES:
            if size * concurrency > MAX_AGGREGATE_PROMPT_TOKENS:
                print(
                    f"skip {size} x C{concurrency}: exceeds aggregate prompt cap",
                    flush=True,
                )
                continue
            case_file = FOLDER / f"{size}-c{concurrency}.json"
            if SKIP_EXISTING and case_file.exists():
                try:
                    existing = json.loads(case_file.read_text())
                except Exception:  # noqa: BLE001 - a corrupt case is re-run
                    existing = []
                if existing and all(
                    not record.get("error")
                    and record.get("output_tokens") == OUTPUT_TOKENS
                    for record in existing
                ):
                    print(f"reuse {case_file.name}", flush=True)
                    summaries.append(summarise(size, concurrency, existing, identity))
                    (FOLDER / "summary.json").write_text(json.dumps(summaries, indent=2))
                    (FOLDER / "TABLE.md").write_text(render(summaries, OUTPUT_TOKENS))
                    continue
            print(f"wave input={size} C={concurrency}", flush=True)
            start = time.monotonic()
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                records = list(pool.map(
                    lambda index: stream_one(size, index, start),
                    range(concurrency),
                ))
            for record in records:
                record["manifest_sha256"] = identity
                record["tokenizer_sha256"] = TOKENIZER_SHA256
            (FOLDER / f"{size}-c{concurrency}.json").write_text(json.dumps(records))
            summary = summarise(size, concurrency, records, identity)
            summaries.append(summary)
            (FOLDER / "summary.json").write_text(json.dumps(summaries, indent=2))
            (FOLDER / "TABLE.md").write_text(render(summaries, OUTPUT_TOKENS))
            print(json.dumps(summary), flush=True)
            if summary["successes"] != concurrency and os.environ.get("CONTINUE_ON_ERROR") != "1":
                print("wave failed; stopping. Set CONTINUE_ON_ERROR=1 to continue.", flush=True)
                return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
