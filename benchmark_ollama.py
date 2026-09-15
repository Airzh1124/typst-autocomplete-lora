#!/usr/bin/env python3
"""Benchmark Ollama with the exact Continue-shaped Qwen FIM prompt."""
from __future__ import annotations

import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def request(base_url: str, model: str, prompt: str, max_tokens: int, keep_alive: str) -> tuple[float, float, str]:
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "raw": True,
        "stream": True,
        "keep_alive": keep_alive,
        "options": {
            "temperature": 0,
            "num_predict": max_tokens,
            "stop": ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|fim_pad|>"],
        },
    }).encode()
    started = time.perf_counter()
    first_token = None
    output: list[str] = []
    with urllib.request.urlopen(
        urllib.request.Request(f"{base_url.rstrip('/')}/api/generate", data=payload, headers={"Content-Type": "application/json"}),
        timeout=180,
    ) as response:
        for line in response:
            event = json.loads(line)
            text = event.get("response", "")
            if text and first_token is None:
                first_token = time.perf_counter()
            output.append(text)
            if event.get("done"):
                break
    finished = time.perf_counter()
    return (first_token or finished) - started, finished - started, "".join(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="qwen2.5-coder-typst:7b")
    parser.add_argument("--data", type=Path, default=Path("data/fim/test.jsonl"))
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--output", type=Path, default=Path("artifacts/ollama-benchmark.json"))
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.open(encoding="utf-8")][: args.limit]
    prompts = [f"<|fim_prefix|>{row['prefix']}<|fim_suffix|>{row['suffix']}<|fim_middle|>" for row in rows]
    if not prompts:
        raise ValueError("No prompts")
    # Load once, then discard the warm-up sample from percentile statistics.
    request(args.base_url, args.model, prompts[0], args.max_tokens, "30m")
    measurements = []
    for prompt in prompts:
        ttft, total, output = request(args.base_url, args.model, prompt, args.max_tokens, "30m")
        measurements.append({"ttft_seconds": ttft, "total_seconds": total, "output": output})
    ttft = [row["ttft_seconds"] for row in measurements]
    total = [row["total_seconds"] for row in measurements]
    result = {
        "model": args.model,
        "examples": len(measurements),
        "prompt_source": str(args.data),
        "prompt_tokens": 2048,
        "warmup_excluded": True,
        "ttft_seconds": {"p50": statistics.median(ttft), "p95": percentile(ttft, 0.95)},
        "total_seconds": {"p50": statistics.median(total), "p95": percentile(total, 0.95)},
        "measurements": measurements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "measurements"}, indent=2))


if __name__ == "__main__":
    main()
