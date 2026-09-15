#!/usr/bin/env python3
"""Turn ignored Typst source splits into Qwen PSM/FIM training examples."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
EOS = "<|endoftext|>"


def token_ids(tokenizer: Any, text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def clip_prefix(tokenizer: Any, text: str, limit: int) -> str:
    ids = token_ids(tokenizer, text)
    return tokenizer.decode(ids[-limit:], skip_special_tokens=False) if limit else ""


def clip_suffix(tokenizer: Any, text: str, limit: int) -> str:
    ids = token_ids(tokenizer, text)
    return tokenizer.decode(ids[:limit], skip_special_tokens=False) if limit else ""


def line_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line in text.splitlines(keepends=True):
        end = start + len(line)
        spans.append((start, end))
        start = end
    if start < len(text) or not spans:
        spans.append((start, len(text)))
    return spans


def make_example(
    tokenizer: Any,
    text: str,
    start: int,
    end: int,
    sample_id: str,
    kind: str,
    max_prompt_tokens: int = 2048,
    max_target_tokens: int = 128,
) -> dict[str, Any] | None:
    middle = text[start:end]
    middle_ids = token_ids(tokenizer, middle)
    if not (2 <= len(middle_ids) <= max_target_tokens):
        return None

    prefix_budget = int(max_prompt_tokens * 0.60)
    suffix_budget = int(max_prompt_tokens * 0.20)
    prefix = clip_prefix(tokenizer, text[:start], prefix_budget)
    suffix = clip_suffix(tokenizer, text[end:], suffix_budget)
    prefix_ids = token_ids(tokenizer, prefix)
    suffix_ids = token_ids(tokenizer, suffix)
    special = [
        tokenizer.convert_tokens_to_ids(FIM_PREFIX),
        tokenizer.convert_tokens_to_ids(FIM_SUFFIX),
        tokenizer.convert_tokens_to_ids(FIM_MIDDLE),
        tokenizer.convert_tokens_to_ids(EOS),
    ]
    available = max_prompt_tokens - len(special) - len(middle_ids)
    if available < 0:
        return None
    # Preserve the Continue ratio first, then spend any remaining budget on
    # prefix context.  Suffix remains capped because cursor completion benefits
    # from a short look-ahead and this matches the existing Continue config.
    if len(prefix_ids) + len(suffix_ids) > available:
        prefix_ids = prefix_ids[-max(0, available - len(suffix_ids)) :]
        if len(prefix_ids) + len(suffix_ids) > available:
            suffix_ids = suffix_ids[: max(0, available - len(prefix_ids))]
        prefix = tokenizer.decode(prefix_ids, skip_special_tokens=False)
        suffix = tokenizer.decode(suffix_ids, skip_special_tokens=False)

    input_ids = (
        [special[0]]
        + token_ids(tokenizer, prefix)
        + [special[1]]
        + token_ids(tokenizer, suffix)
        + [special[2]]
        + middle_ids
        + [special[3]]
    )
    labels = [-100] * (len(input_ids) - len(middle_ids) - 1) + middle_ids + [special[3]]
    assert len(input_ids) == len(labels)
    assert input_ids[:1] == [special[0]]
    assert input_ids[-1:] == [special[3]]
    return {
        "id": sample_id,
        "kind": kind,
        "source_start": start,
        "source_end": end,
        "prefix": prefix,
        "suffix": suffix,
        "middle": middle,
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


def candidate_spans(text: str, rng: random.Random, kind: str, attempts: int = 80):
    spans = line_spans(text)
    if not spans:
        return
    for _ in range(attempts):
        if kind == "inline":
            line_start, line_end = rng.choice(spans)
            if line_end - line_start < 3:
                continue
            # Leave at least two characters for a meaningful completion.
            start = rng.randint(line_start, max(line_start, line_end - 2))
            end = line_end
        else:
            first = rng.randrange(len(spans))
            count = rng.randint(1, min(5, len(spans) - first))
            start, end = spans[first][0], spans[first + count - 1][1]
        if end > start:
            yield start, end


def process_split(
    tokenizer: Any,
    source: Path,
    output: Path,
    split: str,
    seed: int,
    samples_per_kind: int,
    max_prompt_tokens: int,
) -> int:
    split_offset = {"train": 0, "validation": 1, "test": 2}[split]
    rng = random.Random(seed + split_offset)
    count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with source.open(encoding="utf-8") as source_handle, output.open("w", encoding="utf-8") as out:
        for row_index, line in enumerate(source_handle):
            row = json.loads(line)
            for kind in ("inline", "block"):
                made = 0
                for attempt, (start, end) in enumerate(candidate_spans(row["content"], rng, kind)):
                    sample = make_example(
                        tokenizer,
                        row["content"],
                        start,
                        end,
                        f"{split}-{row_index}-{kind}-{attempt}",
                        kind,
                        max_prompt_tokens=max_prompt_tokens,
                    )
                    if sample is None:
                        continue
                    sample.update({"repo": row["repo"], "file": row["file"], "license": row["license"]})
                    out.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    count += 1
                    made += 1
                    if made >= samples_per_kind:
                        break
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-Coder-7B")
    parser.add_argument("--input-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/fim"))
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--samples-per-kind", type=int, default=1)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    for split in ("train", "validation", "test"):
        count = process_split(
            tokenizer,
            args.input_dir / f"{split}.jsonl",
            args.output_dir / f"{split}.jsonl",
            split,
            args.seed,
            args.samples_per_kind,
            args.max_prompt_tokens,
        )
        print(f"{split}: {count} samples")


if __name__ == "__main__":
    main()
