#!/usr/bin/env python3
"""Download and filter Typst-Train without writing source text to Git.

The output JSONL is intentionally ignored by .gitignore.  The audit JSON only
contains counts and aggregate size statistics, not source snippets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlparse

from datasets import load_dataset

DATASET = "TechxGenus/Typst-Train"
REVISION = "a2c78228d501cfdcea8db923531a5b7341d992f4"
ALLOWED_LICENSES = {
    "MIT License",
    "MIT No Attribution",
    "Apache License 2.0",
    "BSD 2-Clause \"Simplified\" License",
    "BSD 3-Clause \"New\" or \"Revised\" License",
    "BSD Zero Clause License",
    "ISC License",
    "zlib License",
    "Creative Commons Zero v1.0 Universal",
    "The Unlicense",
}

# Conservative exclusions: code in test fixtures and generated output is a
# poor source for editor completion and often contains intentionally invalid
# Typst.  A future dataset revision can add explicit allowlists if needed.
EXCLUDED_PATH = re.compile(
    r"(?:^|/)(?:tests?|testdata|fixtures?|generated|dist|build|target|node_modules)(?:/|$)",
    re.IGNORECASE,
)
EXCLUDED_NAME = re.compile(
    r"(?:\.generated\.|\.min\.|_generated\.|generated_)|(?:lock\.typ$)",
    re.IGNORECASE,
)


def normalize_license(value: str | None) -> str:
    return " ".join((value or "").split())


def path_from_url(value: str) -> str:
    return urlparse(value).path.lstrip("/") if value.startswith("http") else value


def is_allowed(row: dict, max_chars: int) -> tuple[bool, str]:
    if row.get("language") != "typst":
        return False, "not_typst"
    license_name = normalize_license(row.get("license"))
    if license_name not in ALLOWED_LICENSES:
        return False, "license"
    file_path = path_from_url(str(row.get("file", "")))
    if EXCLUDED_PATH.search(file_path) or EXCLUDED_NAME.search(file_path):
        return False, "generated_or_test"
    content = row.get("content")
    if not isinstance(content, str) or not content.strip():
        return False, "empty"
    if len(content) > max_chars:
        return False, "too_large"
    return True, "kept"


def split_name(repo: str) -> str:
    bucket = int(hashlib.sha256(repo.encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 90:
        return "train"
    if bucket < 95:
        return "validation"
    return "test"


def write_splits(rows: list[dict], output_dir: Path) -> dict[str, int]:
    grouped: dict[str, list[dict]] = {name: [] for name in ("train", "validation", "test")}
    repo_splits: dict[str, str] = {}
    for row in rows:
        split = split_name(row["repo"])
        previous = repo_splits.setdefault(row["repo"], split)
        if previous != split:
            raise AssertionError(f"repo crossed split boundary: {row['repo']}")
        grouped[split].append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, split_rows in grouped.items():
        with (output_dir / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return {split: len(split_rows) for split, split_rows in grouped.items()}


def aggregate(rows: list[dict], reasons: Counter[str], split_counts: dict[str, int]) -> dict:
    lengths = [len(row["content"]) for row in rows]
    licenses = Counter(row["license"] for row in rows)
    repos = {row["repo"] for row in rows}
    return {
        "dataset": DATASET,
        "revision": REVISION,
        "kept_records": len(rows),
        "kept_repositories": len(repos),
        "split_counts": split_counts,
        "split_repositories": {
            split: len({row["repo"] for row in rows if split_name(row["repo"]) == split})
            for split in ("train", "validation", "test")
        },
        "kept_characters": sum(lengths),
        "kept_bytes_utf8": sum(len(row["content"].encode("utf-8")) for row in rows),
        "content_chars": {
            "min": min(lengths, default=0),
            "max": max(lengths, default=0),
            "mean": round(sum(lengths) / len(lengths), 2) if lengths else 0,
        },
        "license_counts": dict(sorted(licenses.items())),
        "filter_counts": dict(sorted(reasons.items())),
        "content_sha256": hashlib.sha256(
            "".join(sorted(row["content"] for row in rows)).encode("utf-8")
        ).hexdigest(),
    }


def load_local_rows(source: Path) -> list[dict]:
    if source.is_dir():
        candidates = [source / "typst_train.json", source / "data" / "typst_train.json"]
        source = next((path for path in candidates if path.is_file()), source)
    if source.is_dir():
        raise FileNotFoundError(f"Could not find typst_train.json under {source}")
    if source.suffix == ".jsonl":
        return [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    value = json.loads(source.read_text(encoding="utf-8"))
    if isinstance(value, dict) and "train" in value:
        value = value["train"]
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list in {source}")
    return value


def load_rows(source: str, revision: str) -> list[dict]:
    local = Path(source).expanduser()
    if local.exists():
        return load_local_rows(local)
    return list(load_dataset(source, revision=revision, split="train"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=DATASET, help="HF dataset ID or local typst_train.json/.jsonl")
    parser.add_argument("--output", type=Path, default=Path("data/filtered.jsonl"))
    parser.add_argument("--audit", type=Path, default=Path("data/audit.json"))
    parser.add_argument("--max-chars", type=int, default=200_000)
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--revision", default=REVISION)
    args = parser.parse_args()
    if args.revision != REVISION:
        raise ValueError(f"Expected pinned revision {REVISION}, got {args.revision}")

    dataset = load_rows(args.source, args.revision)
    kept: list[dict] = []
    reasons: Counter[str] = Counter()
    seen_content: set[str] = set()
    for row in dataset:
        ok, reason = is_allowed(row, args.max_chars)
        if not ok:
            reasons[reason] += 1
            continue
        content_hash = hashlib.sha256(row["content"].encode("utf-8")).hexdigest()
        if content_hash in seen_content:
            reasons["duplicate_content"] += 1
            continue
        seen_content.add(content_hash)
        kept.append(
            {
                "repo": row["repo"],
                "file": row["file"],
                "license": normalize_license(row["license"]),
                "content": row["content"],
                "content_sha256": content_hash,
            }
        )
        reasons["kept"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in kept:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    split_counts = write_splits(kept, args.split_dir)
    summary = aggregate(kept, reasons, split_counts)
    args.audit.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "split_dir": str(args.split_dir), "audit": str(args.audit), **summary}, indent=2))


if __name__ == "__main__":
    main()
