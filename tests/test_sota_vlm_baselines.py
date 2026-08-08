from __future__ import annotations

import unittest

from scripts.evaluate_sota_vlm_baselines import balanced_indices, extract_move_sequence, parse_letter, parse_token, summarize_task


class SotaVlmBaselineTests(unittest.TestCase):
    def test_strict_parsers(self) -> None:
        self.assertEqual(parse_token("DIFFERENT", ("SAME", "DIFFERENT")), "DIFFERENT")
        self.assertEqual(parse_token("Answer: same.", ("SAME", "DIFFERENT")), "SAME")
        self.assertEqual(parse_letter("(B)", 2), 1)
        self.assertEqual(parse_letter("The answer is C.", 4), 2)
        self.assertIsNone(parse_letter("unclear", 4))

    def test_move_parser_does_not_extract_prose_letters(self) -> None:
        self.assertEqual(extract_move_sequence("RRDDLU"), "RRDDLU")
        self.assertEqual(extract_move_sequence("The route is RRDDLU."), "RRDDLU")
        self.assertEqual(extract_move_sequence("I cannot determine it."), "")

    def test_balanced_indices(self) -> None:
        labels = [0, 0, 0, 1, 1, 1]
        indices = balanced_indices(labels, 4)
        self.assertEqual(indices, [0, 1, 3, 4])
        self.assertEqual([labels[index] for index in indices], [0, 0, 1, 1])

    def test_sat_circular_summary_clusters_answer_orders(self) -> None:
        rows = [
            {"item_id": "0", "base_item_id": "0", "question_type": "ego", "correct": True, "invalid": False},
            {"item_id": "0:reversed", "base_item_id": "0", "question_type": "ego", "correct": False, "invalid": False},
            {"item_id": "1", "base_item_id": "1", "question_type": "ego", "correct": True, "invalid": False},
            {"item_id": "1:reversed", "base_item_id": "1", "question_type": "ego", "correct": True, "invalid": False},
        ]
        result = summarize_task("sat_v2", rows)
        self.assertEqual(result["n"], 2)
        self.assertEqual(result["presentations"], 4)
        self.assertEqual(result["value"], 0.75)
        self.assertEqual(result["ci_method"], "item_cluster_bootstrap_10000")


if __name__ == "__main__":
    unittest.main()
