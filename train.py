#!/usr/bin/env python3
"""Train a Qwen2.5-Coder FIM adapter with 4-bit QLoRA."""
from __future__ import annotations

import argparse
import json
import tomllib
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainingArguments

TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def load_jsonl(path: Path, limit: int | None = None, max_length: int | None = None) -> Dataset:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if max_length and len(row["input_ids"]) > max_length:
                continue
            rows.append({key: row[key] for key in ("input_ids", "labels", "attention_mask")})
            if limit and len(rows) >= limit:
                break
    if not rows:
        raise ValueError(f"No examples in {path}")
    return Dataset.from_list(rows)


def collate(features: list[dict[str, list[int]]], pad_token_id: int) -> dict[str, torch.Tensor]:
    length = max(len(row["input_ids"]) for row in features)
    return {
        "input_ids": torch.tensor([row["input_ids"] + [pad_token_id] * (length - len(row["input_ids"])) for row in features]),
        "labels": torch.tensor([row["labels"] + [-100] * (length - len(row["labels"])) for row in features]),
        "attention_mask": torch.tensor([row["attention_mask"] + [0] * (length - len(row["attention_mask"])) for row in features]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/qlora-7b.toml"))
    parser.add_argument("--resume-from-checkpoint")
    args = parser.parse_args()
    config = tomllib.loads(args.config.read_text(encoding="utf-8"))
    model_cfg, data_cfg, run = config["model"], config["data"], config["training"]
    model_path = Path(model_cfg["path"])
    if not (model_path / "model.safetensors.index.json").is_file():
        raise FileNotFoundError(f"Local model is incomplete: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        quantization_config=quantization,
        device_map={"": 0},
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    # ponytail: keep frozen norms in BF16 on the 12 GB laptop; use PEFT's
    # prepare_model_for_kbit_training (FP32 norms) when >=16 GB is available.
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(
        model,
        LoraConfig(
            task_type="CAUSAL_LM",
            base_model_name_or_path=model_cfg["id"],
            revision=model_cfg["revision"],
            r=model_cfg["lora_rank"],
            lora_alpha=model_cfg["lora_alpha"],
            lora_dropout=model_cfg["lora_dropout"],
            bias="none",
            target_modules=TARGET_MODULES,
        ),
    )
    model.config.use_cache = False
    model.print_trainable_parameters()
    torch.cuda.reset_peak_memory_stats()

    train = load_jsonl(
        Path(data_cfg["train"]), data_cfg.get("max_train_samples"), data_cfg.get("max_sequence_length")
    )
    validation = load_jsonl(
        Path(data_cfg["validation"]), data_cfg.get("max_eval_samples"), data_cfg.get("max_sequence_length")
    )
    output = Path(run["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    training_args: dict[str, Any] = {
        "output_dir": str(output),
        "do_train": True,
        "do_eval": True,
        "per_device_train_batch_size": 1,
        "per_device_eval_batch_size": 1,
        "gradient_accumulation_steps": run["gradient_accumulation_steps"],
        "learning_rate": run["learning_rate"],
        "num_train_epochs": run["epochs"],
        "max_steps": run.get("max_steps", -1),
        "warmup_steps": run.get("warmup_steps", 0),
        "optim": "paged_adamw_8bit",
        "bf16": True,
        "gradient_checkpointing": True,
        "gradient_checkpointing_kwargs": {"use_reentrant": False},
        "eval_strategy": "steps",
        "eval_steps": run["eval_steps"],
        "save_strategy": "steps",
        "save_steps": run["eval_steps"],
        "save_total_limit": 2,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "logging_steps": run["logging_steps"],
        "logging_first_step": True,
        "report_to": "none",
        "seed": run["seed"],
        "data_seed": run["seed"],
        "remove_unused_columns": False,
    }
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**training_args),
        train_dataset=train,
        eval_dataset=validation,
        data_collator=lambda rows: collate(rows, tokenizer.pad_token_id),
        processing_class=tokenizer,
    )
    result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    trainer.save_model(output / "adapter")
    tokenizer.save_pretrained(output / "adapter")
    metrics = {
        **result.metrics,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "peak_cuda_memory_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }
    (output / "train_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
