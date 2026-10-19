#!/usr/bin/env python3
"""Bounded feature acceptance for the GLM-5.3-Flash FP4 recipe.

Run only when the benchmark matrix is not running.  Every check is saved to
`results/feature-acceptance.json`; the needle check also gets its own receipt.
These are smoke checks, not quality scores.
"""

from __future__ import annotations

import base64
import glob
import hashlib
import json
import os
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
import uuid
import zlib

from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ.get("RESULTS_DIR", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
API = os.environ.get("API_BASE", "http://127.0.0.1:8000").rstrip("/")
MODEL = os.environ.get("MODEL", "nvidia/GLM-5.3-Flash-FP4-Local")
NEEDLE = os.environ.get("NEEDLE", "violet-harbor-7291")
NEEDLE_TOKENS = int(os.environ.get("NEEDLE_TOKENS", "190000"))

tokenizer_file = Path(os.environ.get("TOKENIZER_PATH") or sorted(glob.glob(
    str(Path.home() / ".cache/huggingface/hub/"
        "models--nvidia--GLM-5.3-Flash-FP4-Local/snapshots/*/tokenizer.json")))[-1])
TOK = Tokenizer.from_file(str(tokenizer_file))


def post(path: str, body: dict, timeout: float = 600) -> tuple[dict, float]:
    request = urllib.request.Request(
        API + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    start = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read()), time.time() - start


def get(path: str, timeout: float = 60) -> tuple[dict, float]:
    start = time.time()
    with urllib.request.urlopen(API + path, timeout=timeout) as response:
        return json.loads(response.read()), time.time() - start


def chat(messages: list[dict], **extra):
    body = {
        "model": MODEL,
        "messages": messages,
        "temperature": 0,
        "max_tokens": 512,
        "chat_template_kwargs": {"reasoning_effort": "low"},
    }
    body.update(extra)
    return post("/v1/chat/completions", body)


def png(w: int, h: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw))
            + chunk(b"IEND", b""))


def needle_prompt(tokens: int) -> list[int]:
    filler = TOK.encode(
        "The archive records routine maintenance, inventory counts and shipping "
        "manifests. Each entry is ordinary and contains no code.\n",
        add_special_tokens=False,
    ).ids
    secret = TOK.encode(
        f"Remember this: the secret passphrase is {NEEDLE}.\n",
        add_special_tokens=False,
    ).ids
    question = TOK.encode(
        "\nWhat is the secret passphrase mentioned in the text above? "
        "Reply with the passphrase only.\n",
        add_special_tokens=False,
    ).ids
    prefix = TOK.encode(
        f"session {uuid.uuid4().hex}\nRead the archive.\n", add_special_tokens=False
    ).ids
    room = tokens - len(prefix) - len(secret) - len(question)
    body = (filler * (room // len(filler) + 1))[:room]
    at = int(len(body) * 0.6)
    ids = prefix + body[:at] + secret + body[at:] + question
    assert len(ids) == tokens
    return ids


def run() -> dict:
    checks: list[dict] = []

    def record(name: str, ok: bool, detail: str, seconds: float | None = None) -> None:
        checks.append({"name": name, "status": "pass" if ok else "fail",
                       "detail": detail, "seconds": seconds})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {detail}", flush=True)

    try:
        start = time.time()
        with urllib.request.urlopen(API + "/health", timeout=30) as response:
            status = response.status
        record("health", status == 200, f"HTTP {status}", time.time() - start)
    except Exception as error:  # noqa: BLE001
        record("health", False, repr(error))

    try:
        models, seconds = get("/v1/models")
        ids = [item["id"] for item in models.get("data", [])]
        record("models", MODEL in ids, f"served ids: {ids}", seconds)
    except Exception as error:  # noqa: BLE001
        record("models", False, repr(error))

    try:
        reply, seconds = chat(
            [{"role": "user", "content": "What is 19 + 23? Answer with only the number."}],
            max_tokens=128,
        )
        content = reply["choices"][0]["message"].get("content") or ""
        record("arithmetic_19_plus_23", content.strip().endswith("42"),
               f"content={content!r}", seconds)
    except Exception as error:  # noqa: BLE001
        record("arithmetic_19_plus_23", False, repr(error))

    schema = {
        "type": "object",
        "properties": {
            "city": {"type": "string"},
            "temperature_c": {"type": "integer"},
            "conditions": {"type": "string"},
        },
        "required": ["city", "temperature_c", "conditions"],
        "additionalProperties": False,
    }
    try:
        reply, seconds = chat(
            [{"role": "user", "content": "Return the current weather for Paris as JSON."}],
            max_tokens=256,
            response_format={"type": "json_schema",
                             "json_schema": {"name": "weather", "strict": True,
                                             "schema": schema}},
        )
        content = reply["choices"][0]["message"].get("content") or ""
        parsed = json.loads(content)
        ok = set(schema["required"]) <= set(parsed)
        record("json_schema", ok, f"content={content!r}", seconds)
    except Exception as error:  # noqa: BLE001
        record("json_schema", False, repr(error))

    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        },
    }]
    try:
        first, seconds = chat(
            [{"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."}],
            max_tokens=256, tools=tools, tool_choice="auto",
        )
        message = first["choices"][0]["message"]
        calls = message.get("tool_calls") or []
        call_ok = bool(calls) and calls[0]["function"]["name"] == "get_weather"
        record("tool_call_request", call_ok, json.dumps(calls)[:500], seconds)
        if call_ok:
            call = calls[0]
            second, seconds = chat(
                [
                    {"role": "user", "content": "What is the weather in Paris? Use the get_weather tool."},
                    {"role": "assistant", "content": None, "tool_calls": calls},
                    {"role": "tool", "tool_call_id": call["id"],
                     "content": '{"temperature_c": 22, "conditions": "sunny"}'},
                ],
                max_tokens=256, tools=tools,
            )
            content = second["choices"][0]["message"].get("content") or ""
            record("tool_call_round_trip", bool(content.strip()),
                   f"content={content!r}", seconds)
        else:
            record("tool_call_round_trip", False, "no tool call to complete")
    except Exception as error:  # noqa: BLE001
        record("tool_call_request", False, repr(error))
        record("tool_call_round_trip", False, "skipped after request failure")

    try:
        image = base64.b64encode(png(64, 64, (220, 20, 20))).decode()
        reply, seconds = chat(
            [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": "data:image/png;base64," + image}},
                {"type": "text", "text": "What color is this square? Answer with one word."},
            ]}],
            max_tokens=64,
        )
        content = reply["choices"][0]["message"].get("content") or ""
        record("vision_single_red_square", "red" in content.lower(),
               f"content={content!r}", seconds)
    except Exception as error:  # noqa: BLE001
        record("vision_single_red_square", False, repr(error))

    try:
        reply, seconds = post(
            "/v1/messages/count_tokens",
            {"model": MODEL, "messages": [{"role": "user", "content": "Hello world"}]},
            timeout=60,
        )
        record("anthropic_count_tokens", reply.get("input_tokens", 0) > 0,
               json.dumps(reply), seconds)
    except Exception as error:  # noqa: BLE001
        record("anthropic_count_tokens", False, repr(error))

    try:
        reply, seconds = post(
            "/v1/messages",
            {"model": MODEL, "max_tokens": 256,
             "messages": [{"role": "user", "content": "Say hello in one word."}]},
            timeout=300,
        )
        record("anthropic_messages", bool(reply.get("content")),
               json.dumps(reply.get("content"))[:500], seconds)
    except Exception as error:  # noqa: BLE001
        record("anthropic_messages", False, repr(error))

    # Exact-token prompt reuse: identical token IDs, two requests. The second
    # should hit the prefix cache and have a much lower TTFT.
    try:
        ids = needle_prompt(32768)
        body = {
            "model": MODEL, "prompt": ids, "max_tokens": 16,
            "temperature": 0, "stream": False, "ignore_eos": True,
        }
        before = urllib.request.urlopen(API + "/metrics", timeout=20).read().decode()
        hits_before = float(next(
            line.split()[-1] for line in before.splitlines()
            if line.startswith("vllm:prefix_cache_hits_total")))
        _, first_s = post("/v1/completions", body, timeout=600)
        after = urllib.request.urlopen(API + "/metrics", timeout=20).read().decode()
        hits_after = float(next(
            line.split()[-1] for line in after.splitlines()
            if line.startswith("vllm:prefix_cache_hits_total")))
        _, second_s = post("/v1/completions", body, timeout=600)
        record("prompt_reuse_32k", second_s < first_s and hits_after >= hits_before,
               f"first={first_s:.2f}s second={second_s:.2f}s "
               f"cache_hits={hits_before}->{hits_after}", second_s)
    except Exception as error:  # noqa: BLE001
        record("prompt_reuse_32k", False, repr(error))

    try:
        ids = needle_prompt(NEEDLE_TOKENS)
        # The raw completion path tends to continue the reference text instead of
        # answering the question. Use the chat template for a clean retrieval
        # answer; the token-controlled prompt still fixes the haystack length.
        text = TOK.decode(ids)
        reply, seconds = chat(
            [{"role": "user", "content": text}], max_tokens=512
        )
        content = reply["choices"][0]["message"].get("content") or ""
        usage = reply.get("usage", {})
        ok = NEEDLE in content and len(content) < 300
        receipt = {
            "kind": "needle",
            "requested_tokens": NEEDLE_TOKENS,
            "method": "chat",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "seconds": seconds,
            "answer": content,
            "correct": ok,
            "model": MODEL,
            "tokenizer_sha256": hashlib.sha256(tokenizer_file.read_bytes()).hexdigest(),
        }
        (RESULTS / "needle-190k.json").write_text(json.dumps(receipt, indent=2) + "\n")
        record("needle_190k", ok, f"prompt_tokens={usage.get('prompt_tokens')} "
               f"answer={content!r}", seconds)
    except Exception as error:  # noqa: BLE001
        record("needle_190k", False, repr(error))

    result = {
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "model": MODEL,
        "api": API,
        "checks": checks,
        "passed": sum(1 for check in checks if check["status"] == "pass"),
        "total": len(checks),
    }
    (RESULTS / "feature-acceptance.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"{result['passed']}/{result['total']} checks passed")
    return result


if __name__ == "__main__":
    run()
