# Qwen2.5-Coder Typst Autocomplete LoRA

## Summary

A LoRA adapter for Qwen2.5-Coder-7B base, trained for Typst fill-in-the-middle completion. The intended client is Continue in VS Code through Ollama.

## Base model and license

- Base: `Qwen/Qwen2.5-Coder-7B`
- Base revision: `0396a76181e127dfc13e5c5ec48a8cee09938b02`
- Base license: Apache-2.0
- Adapter license: Apache-2.0

## Intended use

Use for local Typst code and document completion. The model is not a Typst compiler, formatter, language server, chat assistant, or safety filter. Keep Tinymist diagnostics enabled and review generated text before accepting it.

## Training

- Objective: Qwen PSM/FIM prompt with loss only on the middle and EOS tokens.
- Data: explicitly permissive-license `.typ` files from `TechxGenus/Typst-Train`, filtered locally; no source text or personal documents are redistributed.
- Split: exact content SHA-256 deduplication followed by repo-level deterministic 90/5/5 split.
- Counts: 12,270 training, 518 validation, and 432 test examples.
- Method: 4-bit NF4 QLoRA, double quantization, BF16 compute, gradient checkpointing, and paged 8-bit AdamW.
- LoRA: rank 16, alpha 32, dropout 0.05; attention and MLP projections.
- Hardware target: RTX 5070 Ti Laptop 12GB or RTX 4080 Super.

## Unified FIM evaluation

The public comparison uses all 432 held-out examples with the same raw-text FIM prompt, llama.cpp runtime, context size (4,096), maximum output (128 tokens), temperature (0), and warm-up procedure:

| Model | Base/adapter format | Exact match | Character similarity | TTFT p50 | Generation p50 |
|---|---|---:|---:|---:|---:|
| Qwen2.5-Coder-7B + this LoRA | Q8_0 base + F16 LoRA GGUF | **29.4%** | **0.674** | 0.061s | 0.220s |
| Typer 1.5B | Q8_0 standalone GGUF | 13.7% | 0.376 | 0.017s | 0.197s |

The test set is repository-held out from this project's preprocessing split. There is no guarantee that Typer's training data did not overlap with the same upstream or related source material, so this is a project benchmark rather than an independent leaderboard. The models also differ in parameter count and quantization/runtime characteristics; latency and quality should not be interpreted as controlled hardware-neutral comparisons.

A real-project compile-rate evaluation requires a private manifest and is not included in this release.

## Limitations

- Completion quality depends on prefix/suffix context, Typst package availability, and the user's document style.
- The model can produce invalid Typst or plausible but semantically wrong content.
- A 7B local model may not satisfy strict interactive latency limits; a smaller model can be preferable for deployment.
- Licensing of upstream source files must be independently reviewed before redistributing any derived dataset or model.
