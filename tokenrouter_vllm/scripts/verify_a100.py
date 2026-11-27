#!/usr/bin/env python3
"""Verification script for 8xA100 testing (tokenizer scan + concurrent API)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import time
import urllib.request
from typing import Any

DEFAULT_PROMPTS = [
    "Solve step by step: If 3x + 7 = 22, what is x^2 + 2x?",
    "Compute the sum of all prime numbers strictly between 10 and 30. Show your steps.",
    (
        "A train travels 120 km in 1.5 hours, then 90 km in 1 hour. "
        "What is its average speed?"
    ),
    "Prove whether the sum of two odd integers is always even.",
]


def inspect_tokenizer_delimiters(model_path: str, delimiter: str = "\n\n") -> None:
    """Scan model vocabulary for all tokens containing `delimiter`."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    merged: list[tuple[int, str]] = []
    for tid in range(len(tokenizer)):
        text = tokenizer.decode([tid], skip_special_tokens=False)
        if delimiter in text:
            merged.append((tid, text))
    print(f"[tokenizer] model={model_path}")
    print(
        f"[tokenizer] vocab_size={len(tokenizer)}, "
        f"tokens containing {delimiter!r}: {len(merged)}"
    )
    print(f"[tokenizer] sample (first 20): {merged[:20]}")
    for probe_text in (".\n\n", ":\n\n", ")\n\n", "\n\n", "</think>"):
        ids = tokenizer.encode(probe_text, add_special_tokens=False)
        decoded = [tokenizer.decode([i], skip_special_tokens=False) for i in ids]
        print(f"[tokenizer] encode({probe_text!r}) -> ids={ids}, pieces={decoded}")


def send_chat_request(
    base_url: str,
    model: str,
    prompt: str,
    *,
    max_tokens: int,
    temperature: float,
    req_idx: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300.0) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = time.perf_counter() - t0
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "req_idx": req_idx,
        "elapsed_s": elapsed,
        "finish_reason": choice.get("finish_reason"),
        "completion_tokens": usage.get("completion_tokens"),
        "content": choice["message"].get("content", ""),
    }


def run_concurrent_smoke(
    base_url: str,
    model: str,
    *,
    concurrency: int,
    max_tokens: int,
    temperature: float,
) -> None:
    prompts = [DEFAULT_PROMPTS[i % len(DEFAULT_PROMPTS)] for i in range(concurrency)]
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [
            pool.submit(
                send_chat_request,
                base_url,
                model,
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                req_idx=i,
            )
            for i, prompt in enumerate(prompts)
        ]
        results = [f.result() for f in futures]
    wall_s = time.perf_counter() - t0

    total_tokens = sum(r["completion_tokens"] or 0 for r in results)
    print(
        f"\n[smoke] concurrency={concurrency}, wall={wall_s:.2f}s, "
        f"total_completion_tokens={total_tokens}, "
        f"agg_tok_per_s={total_tokens / max(wall_s, 1e-6):.1f}"
    )
    for r in sorted(results, key=lambda x: x["req_idx"]):
        preview = r["content"].replace("\n", "\\n")[:120]
        print(
            f"  req#{r['req_idx']}: {r['elapsed_s']:.2f}s | "
            f"finish={r['finish_reason']} | toks={r['completion_tokens']} | "
            f"preview={preview!r}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify TokenRouter x vLLM dual-engine deployment on A100."
    )
    parser.add_argument(
        "--check-tokenizer",
        default=None,
        help="Optional HF model path/name to scan for merged '\\n\\n' tokens.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="SLM entry OpenAI server base URL.",
    )
    parser.add_argument(
        "--model",
        default="tokenrouter-glimp",
        help="Served model name exposed by SLM entry engine.",
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Only run tokenizer inspection without sending API requests.",
    )
    args = parser.parse_args()

    if args.check_tokenizer:
        inspect_tokenizer_delimiters(args.check_tokenizer)
    if not args.skip_api:
        run_concurrent_smoke(
            args.base_url,
            args.model,
            concurrency=args.concurrency,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
