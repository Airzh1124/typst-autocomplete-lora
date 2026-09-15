import unittest

from make_fim import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, make_example
from prepare_data import split_name


class FakeTokenizer:
    pad_token_id = 0
    unk_token_id = -1
    special = {
        FIM_PREFIX: 1,
        FIM_SUFFIX: 2,
        FIM_MIDDLE: 3,
        "<|endoftext|>": 4,
    }

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [1000 + ord(char) for char in text]}

    def convert_tokens_to_ids(self, token):
        return self.special[token]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(value - 1000) for value in ids if value >= 1000)


class FimTests(unittest.TestCase):
    def test_repo_split_is_deterministic(self):
        self.assertEqual(split_name("https://github.com/example/repo"), split_name("https://github.com/example/repo"))
        splits = {split_name(f"repo-{index}") for index in range(300)}
        self.assertEqual(splits, {"train", "validation", "test"})

    def test_fim_order_and_middle_only_labels(self):
        tokenizer = FakeTokenizer()
        text = "prefix\nMIDDLE\nsuffix\n"
        sample = make_example(tokenizer, text, 7, 13, "sample", "block")
        self.assertIsNotNone(sample)
        sample = sample or {}
        self.assertEqual(sample["input_ids"][0], tokenizer.special[FIM_PREFIX])
        self.assertIn(tokenizer.special[FIM_SUFFIX], sample["input_ids"])
        self.assertIn(tokenizer.special[FIM_MIDDLE], sample["input_ids"])
        first_label = next(index for index, value in enumerate(sample["labels"]) if value != -100)
        self.assertTrue(all(value == -100 for value in sample["labels"][:first_label]))
        self.assertEqual(sample["middle"], "MIDDLE")
        self.assertEqual(len(sample["input_ids"]), len(sample["labels"]))
        self.assertEqual(sample["labels"][-1], tokenizer.special["<|endoftext|>"])


if __name__ == "__main__":
    unittest.main()
