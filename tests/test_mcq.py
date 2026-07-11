from __future__ import annotations

import unittest

import torch

from cedar_m2.eval.mcq import compute_mcq_scores, label_token_ids, render_qwen_mcq_prompt


class FakeTokenizer:
    def encode(self, label: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return {"A": [2], "B": [4], "C": [6], "AB": [2, 4]}[label]

    def apply_chat_template(self, messages, **kwargs):
        self.template_kwargs = kwargs
        return f"USER:{messages[0]['content']}\nASSISTANT:"


class MCQTest(unittest.TestCase):
    def test_label_ids_must_be_unique_single_tokens(self) -> None:
        tokenizer = FakeTokenizer()
        self.assertEqual(label_token_ids(tokenizer), {"A": 2, "B": 4, "C": 6})
        with self.assertRaisesRegex(ValueError, "not a single token"):
            label_token_ids(tokenizer, labels=("AB",))

    def test_fixed_prompt_disables_thinking(self) -> None:
        tokenizer = FakeTokenizer()
        prompt = render_qwen_mcq_prompt(tokenizer, "Question text")
        self.assertIn("exactly one uppercase letter", prompt)
        self.assertFalse(tokenizer.template_kwargs["enable_thinking"])
        self.assertTrue(tokenizer.template_kwargs["add_generation_prompt"])

    def test_predictions_and_correct_margin(self) -> None:
        logits = torch.zeros(2, 10)
        logits[0, [2, 4, 6]] = torch.tensor([3.0, 1.0, 2.0])
        logits[1, [2, 4, 6]] = torch.tensor([-1.0, 0.0, 4.0])
        scores = compute_mcq_scores(
            logits,
            {"A": 2, "B": 4, "C": 6},
            correct_labels=("A", "B"),
        )

        self.assertEqual(scores[0].predicted_label, "A")
        self.assertTrue(scores[0].correct)
        self.assertAlmostEqual(scores[0].margin, 1.0)
        self.assertEqual(scores[1].predicted_label, "C")
        self.assertFalse(scores[1].correct)
        self.assertAlmostEqual(scores[1].margin, -4.0)


if __name__ == "__main__":
    unittest.main()
