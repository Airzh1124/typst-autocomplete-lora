# Typst Autocomplete LoRA

A local Typst **fill-in-the-middle (FIM) autocomplete adapter** for Qwen2.5-Coder-7B. It is designed for Continue + Ollama and complements Tinymist's deterministic symbol, parameter, file, font, and diagnostic features. It is not a chat model, compiler, formatter, or language server.

The repository contains the training and evaluation code. Model weights, source-derived data, private documents, and per-example outputs are intentionally kept out of GitHub. The released PEFT adapter and GGUF adapter are published separately on Hugging Face.

## Current results

The formal run uses 2 epochs, rank-16 QLoRA, and 12,270 training / 518 validation / 432 test FIM examples. The unified GGUF evaluation uses the same llama.cpp runtime, raw-text FIM protocol, context size, and generation limit for both models:

| Model | Exact match | Character similarity | TTFT p50 | Generation p50 |
|---|---:|---:|---:|---:|
| Qwen2.5-Coder-7B + this LoRA (Q8_0 base) | **29.4%** | **0.674** | 0.061s | 0.220s |
| Typer 1.5B (Q8_0) | 13.7% | 0.376 | 0.017s | 0.197s |

These are FIM metrics, not a general Typst or chat benchmark. A real-project compile-rate evaluation requires a private manifest and is not included in this repository.

## Data

The source is `TechxGenus/Typst-Train` at revision `a2c78228d501cfdcea8db923531a5b7341d992f4`. The local preprocessing pipeline:

- keeps Typst files with explicitly permissive licenses (MIT, Apache-2.0, BSD, ISC, zlib, CC0, or Unlicense);
- removes empty, test, generated, oversized, and duplicate content;
- splits by repository with a deterministic 90/5/5 train/validation/test split;
- creates Qwen PSM/FIM examples with loss only on the missing middle and EOS tokens.

The current filtered aggregate is 6,610 files from 988 repositories. Raw source text and derived JSONL files are ignored and are not redistributed by this repository. Upstream license and provenance review remains the publisher's responsibility.

## Environment

Use any Python 3.11 environment with a matching CUDA PyTorch build. The pinned training/evaluation dependencies are in `requirements.txt`:

```bash
conda create -n typst-autocomplete python=3.11 -y
conda activate typst-autocomplete
python -m pip install torch --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
```

For the optional GGUF comparison on a CUDA 12.4 Linux server, install the prebuilt wheel separately:

```bash
python -m pip install -r requirements-eval-gguf.txt
```

The GGUF requirement is platform-specific and deliberately not part of the main training requirements.

## Download the base model

The training and Transformers evaluator expect the pinned local base model:

```bash
hf download Qwen/Qwen2.5-Coder-7B \
  --revision 0396a76181e127dfc13e5c5ec48a8cee09938b02 \
  --local-dir models/Qwen2.5-Coder-7B
```

To rebuild the data on a new machine:

```bash
hf download TechxGenus/Typst-Train \
  --repo-type dataset \
  --revision a2c78228d501cfdcea8db923531a5b7341d992f4 \
  --local-dir data/raw/Typst-Train
python prepare_data.py
python make_fim.py --tokenizer models/Qwen2.5-Coder-7B
```

Alternatively, `prepare_data.py --source` accepts an existing local JSON/JSONL source. The generated data remains ignored.

## Train

The formal configuration is `configs/qlora-7b.toml`:

```bash
python train.py --config configs/qlora-7b.toml
```

It uses NF4 double quantization, BF16 compute, gradient checkpointing, paged 8-bit AdamW, rank-16 LoRA, and 2 epochs. The adapter and metrics are written under `artifacts/qlora-7b/`, which is ignored.

## Evaluate the trained adapter

The unified GGUF evaluator compares the formal Qwen LoRA with Typer 1.5B on the same FIM test set and the same llama.cpp runtime. It requires a Qwen2.5-Coder-7B base GGUF matching the adapter's architecture and revision:

```bash
python evaluate_fim.py \
  --lora-base-model models/Qwen2.5-Coder-7B-Q8/qwen2.5-coder-7b-q8_0.gguf \
  --lora-adapter artifacts/qlora-7b/typst-adapter-f16.gguf \
  --typer-model models/Typer-1.5B/typer-1.5b-base.Q8_0.gguf \
  --data data/fim/test.jsonl \
  --output artifacts/fim-comparison.json
```

Download the matching non-Instruct Qwen base GGUF before running the comparison:

```bash
hf download ggml-org/Qwen2.5-Coder-7B-Q8_0-GGUF qwen2.5-coder-7b-q8_0.gguf \
  --local-dir models/Qwen2.5-Coder-7B-Q8
```

This comparison reports exact match, character similarity, TTFT, and generation latency. GGUF evaluation does not report Transformers middle loss. Record the Qwen and Typer quantization levels with the results.

For private compile evaluation, create an ignored `private_eval/manifest.jsonl` with rows containing `project_root`, `source_file`, `start`, and `end`, then run:

```bash
python compile_eval.py \
  --adapter artifacts/qlora-7b/adapter \
  --manifest private_eval/manifest.jsonl \
  --output artifacts/private-compile-eval.json
```

The private projects must remain outside GitHub and Hugging Face.

## Local deployment

Convert the PEFT adapter to GGUF with a separate llama.cpp conversion environment. Do not install these converter dependencies into the training or GGUF runtime environment:

```bash
git clone --depth 1 https://github.com/ggml-org/llama.cpp.git ~/llama.cpp
python -m venv ~/.venvs/llama-convert
source ~/.venvs/llama-convert/bin/activate
cd ~/llama.cpp
python -m pip install -r requirements/requirements-convert_lora_to_gguf.txt
cd -
```

The converter only needs the local Hugging Face base-model configuration/tokenizer; the full base weights are not loaded for LoRA conversion:

```bash
python ~/llama.cpp/convert_lora_to_gguf.py \
  artifacts/qlora-7b/adapter \
  --base models/Qwen2.5-Coder-7B \
  --outfile artifacts/qlora-7b/typst-adapter-f16.gguf \
  --outtype f16
```

The conversion environment is separate because llama.cpp's converter installs its own CPU PyTorch/Transformers dependency set. Deactivate it before running training or GGUF inference.

The root `Modelfile` points to this formal adapter and assumes the base Ollama model is already available:

```bash
ollama create qwen2.5-coder-typst:7b -f Modelfile
```

Merge `continue-example.yaml` into `~/.continue/config.yaml` and select the `autocomplete` role. Tinymist should remain enabled as the deterministic first layer.

## Benchmark scope

The public quantitative comparison is limited to FIM-compatible autocomplete models evaluated with the same raw-text FIM protocol. Typer 1.5B is a useful small FIM baseline. Chat/instruction models such as Typst-Coder 9B are not directly comparable under this protocol and are not ranked here.

The 432 test samples are repository-held-out from this project's preprocessing split, but there is no guarantee that a third-party baseline has not seen overlapping source material. Model size, quantization, runtime, and prompt protocol must be reported with any comparison.

## License and non-published files

Code and configuration are Apache-2.0. The repository does not redistribute `Typst-Train`, generated source-derived JSONL, base weights, private projects, checkpoints, or per-example model outputs. See `MODEL_CARD.md` for model-specific usage and limitations.
