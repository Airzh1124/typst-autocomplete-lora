#!/usr/bin/env python3
"""Compare a Qwen GGUF + LoRA adapter with Typer on one FIM test set."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

STOP = ["<|fim_prefix|>", "<|fim_suffix|>", "<|fim_middle|>", "<|endoftext|>", "<|fim_pad|>"]


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * fraction))]


def load_rows(path: Path, limit: int | None) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Test data does not exist: {path}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be positive")
        rows = rows[:limit]
    required = {"prefix", "suffix", "middle"}
    if not rows or any(not required <= row.keys() for row in rows):
        raise ValueError(f"FIM data must contain prefix, suffix, and middle: {path}")
    return rows


def prompt(row: dict[str, Any]) -> str:
    return f"<|fim_prefix|>{row['prefix']}<|fim_suffix|>{row['suffix']}<|fim_middle|>"


def evaluate_model(
    label: str,
    model_path: Path,
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    lora_path: Path | None = None,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Model does not exist: {model_path}")
    if lora_path and not lora_path.is_file():
        raise FileNotFoundError(f"LoRA adapter does not exist: {lora_path}")

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise RuntimeError(
            "GGUF evaluation requires llama-cpp-python; install requirements-eval-gguf.txt"
        ) from exc

    print(f"Loading {label}: {model_path}", flush=True)
    kwargs: dict[str, Any] = {
        "model_path": str(model_path),
        "n_ctx": args.context,
        "n_batch": args.batch,
        "n_gpu_layers": args.gpu_layers,
        "verbose": False,
    }
    if lora_path:
        kwargs["lora_path"] = str(lora_path)
        kwargs["lora_scale"] = 1.0
    model = Llama(**kwargs)
    prompts = [prompt(row) for row in rows]

    # Keep model-loading cost out of per-example latency statistics.
    for _ in model.create_completion(
        prompts[0], max_tokens=args.max_tokens, temperature=0.0, stop=STOP, stream=True, echo=False
    ):
        pass

    measurements = []
    for index, (request, row) in enumerate(zip(prompts, rows), start=1):
        started = time.perf_counter()
        first_token = None
        output_parts: list[str] = []
        for event in model.create_completion(
            request,
            max_tokens=args.max_tokens,
            temperature=0.0,
            stop=STOP,
            stream=True,
            echo=False,
        ):
            text = event["choices"][0].get("text", "")
            if text and first_token is None:
                first_token = time.perf_counter()
            output_parts.append(text)
        finished = time.perf_counter()
        output = "".join(output_parts)
        measurements.append(
            {
                "exact_match": output == row["middle"],
                "char_similarity": SequenceMatcher(None, output, row["middle"]).ratio(),
                "ttft_seconds": (first_token or finished) - started,
                "generation_seconds": finished - started,
                "output": output,
            }
        )
        if index % 25 == 0 or index == len(rows):
            print(f"{label}: {index}/{len(rows)}", flush=True)

    ttft = [item["ttft_seconds"] for item in measurements]
    generation = [item["generation_seconds"] for item in measurements]
    result = {
        "model": str(model_path),
        "lora_adapter": str(lora_path) if lora_path else None,
        "examples": len(measurements),
        "exact_match": statistics.mean(item["exact_match"] for item in measurements),
        "char_similarity": statistics.mean(item["char_similarity"] for item in measurements),
        "ttft_seconds": {"p50": statistics.median(ttft), "p95": percentile(ttft, 0.95)},
        "generation_seconds": {
            "p50": statistics.median(generation),
            "p95": percentile(generation, 0.95),
        },
        "measurements": measurements,
    }
    del model
    gc.collect()
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lora-base-model", type=Path, required=True)
    parser.add_argument("--lora-adapter", type=Path, required=True)
    parser.add_argument("--typer-model", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=Path("data/fim/test.jsonl"))
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=512)
    parser.add_argument("--gpu-layers", type=int, default=-1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/fim-comparison.json"))
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.max_tokens <= 0 or args.context <= args.max_tokens:
        parser.error("--context must be larger than positive --max-tokens")
    if args.batch <= 0:
        parser.error("--batch must be positive")

    rows = load_rows(args.data, args.limit)
    result = {
        "protocol": "raw text Qwen FIM; same llama.cpp runtime and generation settings",
        "data": str(args.data),
        "examples": len(rows),
        "max_tokens": args.max_tokens,
        "context": args.context,
        "models": {},
    }
    result["models"]["qwen_lora"] = evaluate_model(
        "Qwen2.5-Coder-7B + LoRA", args.lora_base_model, rows, args, args.lora_adapter
    )
    result["models"]["typer"] = evaluate_model("Typer 1.5B", args.typer_model, rows, args)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    for label, metrics in result["models"].items():
        print(
            f"{label}: exact={metrics['exact_match']:.3f}, "
            f"similarity={metrics['char_similarity']:.3f}, "
            f"TTFT p50={metrics['ttft_seconds']['p50']:.3f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()
