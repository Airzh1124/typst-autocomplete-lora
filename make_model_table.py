#!/usr/bin/env python3
"""Build a Markdown table from the unified FIM comparison result."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def seconds(value: float) -> str:
    return f"{value:.3f}s"


def render_table(data: dict) -> str:
    rows = data.get("models", {})
    required = {"qwen_lora", "typer"}
    if not required <= rows.keys():
        raise ValueError(f"result must contain models: {', '.join(sorted(required))}")
    lines = [
        "# Typst FIM autocomplete comparison",
        "",
        f"Same {data['examples']} examples, raw-text FIM, llama.cpp, "
        f"context {data['context']}, max tokens {data['max_tokens']}.",
        "",
        "| Model | Exact match | Char similarity | TTFT p50 | Generation p50 |",
        "|---|---:|---:|---:|---:|",
    ]
    for key, name in (("qwen_lora", "Qwen2.5-Coder-7B + LoRA"), ("typer", "Typer 1.5B")):
        row = rows[key]
        lines.append(
            f"| {name} | {pct(row['exact_match'])} | {row['char_similarity']:.3f} | "
            f"{seconds(row['ttft_seconds']['p50'])} | {seconds(row['generation_seconds']['p50'])} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("artifacts/fim-comparison.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/model-comparison.md"))
    args = parser.parse_args()
    table = render_table(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(table, encoding="utf-8")
    print(args.output)
    print(table, end="")


if __name__ == "__main__":
    main()
