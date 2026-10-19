#!/usr/bin/env python3
"""needle.py with a vLLM-compatible stats line: same prompt, seed and passphrase as
tools/needle.py at the TensorFold repo; only drops the TensorFold-only `prefill_s`
field (vLLM has no such field) and prints the client-side round-trip instead.

Usage: needle-vllm.py [label] [size]   (default 248000 -> ~195k prompt tokens).
Env: MODEL, PORT / API_URL (as tools/client.py).
"""
import json
import os
import sys
import time
import urllib.request

sys.dont_write_bytecode = True
sys.path.insert(0, "/home/tngai/data/glm53-flash-tensorfold/tools")
from client import URL, open_url, prose  # noqa: E402

SECRET = "violet-harbor-7291"


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "needle"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 248_000
    hay = prose(size, 777).split(". ")
    at = int(len(hay) * 0.6)
    hay.insert(at, f"Remember this: the secret passphrase is {SECRET}")
    prompt = ". ".join(hay) + "\n\nWhat is the secret passphrase mentioned in the text above? Reply with the passphrase only."
    body = {"model": os.environ.get("MODEL", "GLM-5.3-Flash-EXL3"), "max_tokens": 2048, "temperature": 0, "seed": 1234,
            "messages": [{"role": "user", "content": prompt}]}
    t0 = time.time()
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    reply = json.load(open_url(req, 3600))
    content = reply["choices"][0]["message"].get("content") or ""
    ok = SECRET in content
    tf = (reply.get("tensorfold") or {}).get("prefill_s")
    prefill = f"{tf} s" if tf is not None else "n/a"
    print(f"{label}: needle {reply['usage']['prompt_tokens']} tok, prefill {prefill}, "
          f"roundtrip {time.time() - t0:.1f} s, answer {content.strip()[-80:]!r}: {'CORRECT' if ok else 'WRONG'}", flush=True)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
