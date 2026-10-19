#!/usr/bin/env python3
"""Audit a saved matrix against its launch manifest and summary.

Usage:
  python3 benchmarks/audit.py [results/matrix-<timestamp>]

Writes `protocol-audit.json` next to the matrix (or into RESULTS_DIR).
The audit checks structure, not model quality: exact prompt/output counts,
successful completion, common-window presence, and matching manifest hashes.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))


def main() -> int:
    if len(sys.argv) > 1:
        folder = Path(sys.argv[1])
    else:
        candidates = sorted(glob.glob(str(RESULTS / "matrix-*")))
        if not candidates:
            print("no matrix directories found")
            return 1
        folder = Path(candidates[-1])

    manifest_file = folder / "launch.json"
    if not manifest_file.exists():
        manifest_file = RESULTS / "launch.json"
    manifest_hash = (
        hashlib.sha256(manifest_file.read_bytes()).hexdigest()
        if manifest_file.exists() else None
    )
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    expected_tokenizer = manifest.get("tokenizer_sha256")

    failures: list[str] = []
    warnings: list[str] = []
    cases = sorted(folder.glob("*-c*.json"))
    checked = 0
    for case in cases:
        records = json.loads(case.read_text())
        if not records:
            failures.append(f"{case.name}: empty case")
            continue
        good = [record for record in records if not record.get("error")]
        if len(good) != len(records):
            failures.append(f"{case.name}: {len(records) - len(good)} failed requests")
        for record in records:
            checked += 1
            if record.get("prompt_tokens_reported") != record.get("prompt_tokens_requested"):
                failures.append(
                    f"{case.name}: prompt_tokens "
                    f"{record.get('prompt_tokens_reported')} != "
                    f"{record.get('prompt_tokens_requested')}"
                )
            if record.get("output_tokens") != record.get("output_budget"):
                failures.append(
                    f"{case.name}: output_tokens {record.get('output_tokens')} != "
                    f"{record.get('output_budget')}"
                )
            if record.get("error"):
                failures.append(f"{case.name}: {record['error']}")
            if manifest_hash and record.get("manifest_sha256") not in (None, manifest_hash):
                failures.append(f"{case.name}: manifest hash mismatch")
            if expected_tokenizer and record.get("tokenizer_sha256") not in (
                None, expected_tokenizer
            ):
                failures.append(f"{case.name}: tokenizer hash mismatch")
        if len(good) == len(records) and len(records) > 1:
            begin = max(record["first_token_s"] for record in good)
            end = min(record["finished_s"] for record in good)
            if end <= begin:
                warnings.append(f"{case.name}: no positive common decode window")

    summary_file = folder / "summary.json"
    summary = json.loads(summary_file.read_text()) if summary_file.exists() else []
    if summary:
        if len(summary) != len(cases):
            warnings.append(
                f"summary has {len(summary)} rows for {len(cases)} case files"
            )
        for row in summary:
            if row["successes"] != row["concurrency"]:
                failures.append(
                    f"summary {row['input_tokens']}xC{row['concurrency']}: "
                    f"{row['successes']}/{row['concurrency']} successes"
                )
    else:
        failures.append("summary.json missing or empty")

    audit = {
        "matrix": str(folder),
        "manifest_sha256": manifest_hash,
        "cases": len(cases),
        "requests_checked": checked,
        "failures": failures,
        "warnings": warnings,
        "status": "pass" if not failures else "fail",
    }
    output = RESULTS / "protocol-audit.json"
    output.write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
