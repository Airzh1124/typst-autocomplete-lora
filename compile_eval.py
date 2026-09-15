#!/usr/bin/env python3
"""Compile original and generated holes from a private Typst manifest.

Manifest rows are ignored and never copied into Git:
{"project_root": "/private/project", "source_file": "main.typ",
 "start": 42, "end": 57}
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from make_fim import make_example, token_ids

STOP_IDS = [151643, 151645, 151659, 151660, 151661, 151662, 151663, 151664]


def load_model(path: Path, adapter: Path | None = None):
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        path,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    return PeftModel.from_pretrained(model, adapter, local_files_only=True).eval() if adapter else model.eval()


def predict(model: Any, tokenizer: Any, prompt_ids: list[int], max_new_tokens: int) -> str:
    input_ids = torch.tensor([prompt_ids], device=model.device)
    with torch.inference_mode():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=STOP_IDS,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0, input_ids.shape[1] :], skip_special_tokens=True)


def compile_file(project: Path, relative_file: str, timeout: int = 30) -> bool:
    source = project / relative_file
    output = project / ".autocomplete-compile.pdf"
    try:
        result = subprocess.run(
            ["typst", "compile", str(source), str(output)],
            cwd=project,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        output.unlink(missing_ok=True)
    return result.returncode == 0


def compile_prediction(project_root: Path, relative_file: str, start: int, end: int, replacement: str) -> bool:
    with tempfile.TemporaryDirectory(prefix="typst-autocomplete-") as directory:
        copied = Path(directory) / "project"
        shutil.copytree(project_root, copied)
        source = copied / relative_file
        text = source.read_text(encoding="utf-8")
        source.write_text(text[:start] + replacement + text[end:], encoding="utf-8")
        return compile_file(copied, relative_file)


def rows(path: Path, limit: int | None):
    with path.open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit and index >= limit:
                break
            yield json.loads(line)


def evaluate_model(model: Any, tokenizer: Any, manifest: list[dict[str, Any]], max_new_tokens: int) -> dict[str, Any]:
    attempted = compiled = original_compiled = 0
    for row in manifest:
        root = Path(row["project_root"]).expanduser().resolve()
        relative_file = row["source_file"]
        source = (root / relative_file).resolve()
        if root not in source.parents or not source.is_file():
            raise ValueError(f"source_file must be inside project_root: {source}")
        text = source.read_text(encoding="utf-8")
        example = make_example(tokenizer, text, row["start"], row["end"], "private", "private")
        if example is None:
            continue
        original_compiled += int(compile_prediction(root, relative_file, row["start"], row["end"], example["middle"]))
        prompt_len = example["labels"].index(next(value for value in example["labels"] if value != -100))
        prediction = predict(model, tokenizer, example["input_ids"][:prompt_len], max_new_tokens)
        attempted += 1
        compiled += int(compile_prediction(root, relative_file, row["start"], row["end"], prediction))
    return {
        "examples": attempted,
        "original_compile_rate": original_compiled / len(manifest) if manifest else None,
        "generated_compile_rate": compiled / attempted if attempted else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=Path("private_eval/manifest.jsonl"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen2.5-Coder-7B"))
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--output", type=Path, default=Path("artifacts/private-compile-eval.json"))
    args = parser.parse_args()
    manifest = list(rows(args.manifest, args.limit))
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    base = load_model(args.model)
    base_metrics = evaluate_model(base, tokenizer, manifest, args.max_new_tokens)
    adapted = PeftModel.from_pretrained(base, args.adapter, local_files_only=True).eval()
    adapter_metrics = evaluate_model(adapted, tokenizer, manifest, args.max_new_tokens)
    result = {"manifest": str(args.manifest), "base": base_metrics, "adapter": adapter_metrics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
