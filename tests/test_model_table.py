import unittest

from make_model_table import render_table


class ModelTableTest(unittest.TestCase):
    def test_renders_both_unified_models(self):
        data = {
            "examples": 432,
            "context": 4096,
            "max_tokens": 128,
            "models": {
                "qwen_lora": {
                    "exact_match": 0.293981,
                    "char_similarity": 0.674358,
                    "ttft_seconds": {"p50": 0.0609},
                    "generation_seconds": {"p50": 0.2197},
                },
                "typer": {
                    "exact_match": 0.136574,
                    "char_similarity": 0.375565,
                    "ttft_seconds": {"p50": 0.0172},
                    "generation_seconds": {"p50": 0.1968},
                },
            },
        }
        table = render_table(data)
        self.assertIn("Same 432 examples", table)
        self.assertIn("Qwen2.5-Coder-7B + LoRA | 29.4%", table)
        self.assertIn("Typer 1.5B | 13.7%", table)


if __name__ == "__main__":
    unittest.main()
