from __future__ import annotations

import unittest

from openprism.learning.detection_evaluation import grouped_detection_bootstrap


class DetectionBootstrapTests(unittest.TestCase):
    def test_capture_sequence_bootstrap_is_paired_and_deterministic(self) -> None:
        truth = {
            "190001": [[0.0, 0.0, 10.0, 10.0]],
            "190002": [[2.0, 2.0, 12.0, 12.0]],
            "210001": [[5.0, 5.0, 15.0, 15.0]],
        }
        groups = {sample_id: sample_id[:2] for sample_id in truth}
        perfect = [
            (sample_id, 0.9, boxes[0]) for sample_id, boxes in truth.items()
        ]
        weak = [("190001", 0.8, [20.0, 20.0, 30.0, 30.0])]
        first = grouped_detection_bootstrap(
            truth,
            {"visible_rgb": weak, "candidate": perfect},
            groups,
            replicates=200,
            seed=17,
        )
        second = grouped_detection_bootstrap(
            truth,
            {"visible_rgb": weak, "candidate": perfect},
            groups,
            replicates=200,
            seed=17,
        )

        self.assertEqual(first, second)
        self.assertEqual(first["scene_group_count"], 2)
        comparison = first["paired_deltas_vs_baseline"]["candidate"]
        self.assertGreater(
            comparison["bootstrap_probability_candidate_better"]["ap50"], 0.9
        )
        self.assertGreater(
            comparison["intervals"]["ap50"]["lower"], 0.0
        )

    def test_bootstrap_rejects_missing_group_assignments(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly one scene group"):
            grouped_detection_bootstrap(
                {"a": []},
                {"visible_rgb": []},
                {},
                replicates=10,
            )


if __name__ == "__main__":
    unittest.main()
