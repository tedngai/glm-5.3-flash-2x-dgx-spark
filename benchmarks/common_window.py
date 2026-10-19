#!/usr/bin/env python3
"""Recompute the matrix summary from saved raw token events.

Usage:
  python3 benchmarks/common_window.py results/matrix-20261019T...

Reads every `<size>-c<concurrency>.json` case, rebuilds the common decode
interval, and rewrites `summary.json` and `TABLE.md` in place.  This makes the
published table auditable without rerunning inference.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import statistics
import sys


def summarise(size: int, concurrency: int, records: list[dict]) -> dict:
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
        "manifest_sha256": records[0].get("manifest_sha256") if records else None,
        "tokenizer_sha256": records[0].get("tokenizer_sha256") if records else None,
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
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    folder = Path(sys.argv[1])
    pattern = re.compile(r"^(\d+)-c(\d+)$")
    cases = sorted(
        (p for p in folder.glob("*-c*.json") if pattern.match(p.stem)),
        key=lambda p: tuple(int(x) for x in pattern.match(p.stem).groups()),
    )
    if not cases:
        print(f"no case files in {folder}")
        return 1
    summaries = []
    output_tokens = None
    for case in cases:
        size_text, concurrency_text = case.stem.split("-c")
        records = json.loads(case.read_text())
        if records:
            output_tokens = records[0].get("output_budget", output_tokens)
        summaries.append(summarise(int(size_text), int(concurrency_text), records))
    (folder / "summary.json").write_text(json.dumps(summaries, indent=2))
    (folder / "TABLE.md").write_text(render(summaries, output_tokens or 0))
    print(f"recomputed {len(summaries)} cases in {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
