#!/usr/bin/env python3
"""Indicative decode benchmark for the FP4 vLLM server, mirroring the 1- and 4-request
prose decode figures reported for the TensorFold EXL3 build (61.8 / 103.2 tok/s).

512-token streamed completions; decode tok/s measured between first and last streamed
token. Thinking disabled to match a plain prose workload.
Env: MODEL, PORT / API_URL.
"""
import json
import os
import threading
import time
import urllib.request

API = os.environ.get("API_URL", "http://127.0.0.1:" + os.environ.get("PORT", "8000")).rstrip("/")
URL = API + "/v1/chat/completions"
MODEL = os.environ.get("MODEL", "GLM-5.3-Flash-EXL3")

PROMPT = ("Continue this prose passage naturally, about 400 words, plain prose only:\n\n"
          "The harbor was quiet that morning, gulls tracing slow circles over the water while the last of the "
          "night's fog lifted. A fishing boat eased past the breakwater, its engine a low murmur against the "
          "gulls. On the pier, an old man sorted ropes by habit rather than need, watching the horizon change "
          "from grey to gold.")


EFFORT = os.environ.get("BENCH_EFFORT", "low")


def one(idx: int, results: dict) -> None:
    body = {"model": MODEL, "messages": [{"role": "user", "content": PROMPT}], "max_tokens": 512,
            "temperature": 0.7, "seed": 100 + idx, "stream": True,
            "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"reasoning_effort": EFFORT}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0, t_first, t_last, n, usage = time.time(), None, None, 0, 0
    with urllib.request.urlopen(req, timeout=1800) as r:
        for raw in r:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except ValueError:
                continue
            if ev.get("usage"):
                usage = ev["usage"].get("completion_tokens") or usage
            ch = ev.get("choices") or []
            if not ch:
                continue
            delta = ch[0].get("delta") or {}
            if delta.get("content") or delta.get("reasoning") or delta.get("reasoning_content"):
                now = time.time()
                if t_first is None:
                    t_first = now
                t_last = now
                n += 1
    span = (t_last - t_first) if t_first and t_last and t_last > t_first else None
    tok = usage or n
    results[idx] = {"n": n, "usage": usage, "ttft": (t_first - t0) if t_first else None,
                    "decode": (tok - 1) / span if span and tok > 1 else None,
                    "total": time.time() - t0}


def run(k: int) -> None:
    results: dict = {}
    ths = [threading.Thread(target=one, args=(i, results)) for i in range(k)]
    t0 = time.time()
    for t in ths:
        t.start()
    for t in ths:
        t.join()
    wall = time.time() - t0
    toks = sum((r["usage"] or r["n"]) for r in results.values())
    dec = [r["decode"] for r in results.values() if r["decode"]]
    print(f"streams={k}: per-stream decode {[round(d, 1) for d in dec]} tok/s; "
          f"aggregate {toks / wall:.1f} tok/s over {wall:.1f}s; ttft {[round(r['ttft'], 2) for r in results.values()]}",
          flush=True)


if __name__ == "__main__":
    for streams in (1, 4, 8):
        run(streams)
