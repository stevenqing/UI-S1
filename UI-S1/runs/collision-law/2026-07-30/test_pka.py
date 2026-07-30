import inspect
import math
import unittest

import kernels
import pka
from aggregators import plurality_then_density
from pka import Prediction, coordinate_density_medoid, coordinate_density_mode, pka_joint, pka_joint_continuous


class ProductKernelAggregationTest(unittest.TestCase):
    def test_k1_identity(self):
        prediction = Prediction("click", 0.25, 0.75, source="only")
        result = pka_joint("androidcontrol", [prediction])
        self.assertEqual(result.prediction, prediction)
        self.assertEqual(result.candidate_index, 0)

    def test_parse_failures_are_removed(self):
        bad = Prediction("click", 0.1, 0.1, source="bad", parse_ok=False)
        good = Prediction("click", 0.2, 0.2, source="good")
        result = pka_joint("androidcontrol", [bad, good])
        self.assertEqual(result.prediction, good)
        self.assertEqual(result.candidate_index, 1)

    def test_parameter_free_actions_reduce_to_plurality(self):
        predictions = [
            Prediction("wait", source="a"),
            Prediction("wait", source="b"),
            Prediction("press_back", source="c"),
        ]
        result = pka_joint("androidcontrol", predictions)
        self.assertEqual(result.prediction.action, "wait")
        self.assertEqual(result.candidate_scores, (2.0, 2.0, 1.0))

    def test_same_type_medoid_matches_independent_density_reference(self):
        predictions = [
            Prediction("click", 0.10, 0.10, source="a"),
            Prediction("click", 0.11, 0.10, source="b"),
            Prediction("click", 0.90, 0.90, source="c"),
        ]
        result = pka_joint("androidcontrol", predictions)
        reference_scores = [
            sum(kernels.android_coord_kernel_normalized(v.coordinate, c.coordinate) for v in predictions)
            for c in predictions
        ]
        expected = max(range(len(predictions)), key=lambda index: (reference_scores[index], -index))
        self.assertEqual(result.candidate_index, expected)

    def test_continuous_mode_finds_dense_cluster(self):
        predictions = [
            Prediction("click", 0.10, 0.10, source="a"),
            Prediction("click", 0.11, 0.10, source="b"),
            Prediction("click", 0.90, 0.90, source="c"),
        ]
        coordinate = coordinate_density_mode("androidcontrol", predictions, "click")
        self.assertLess(math.dist(coordinate, (0.105, 0.10)), 0.01)
        result = pka_joint_continuous("androidcontrol", predictions)
        self.assertEqual(result.prediction.action, "click")
        self.assertLess(math.dist(result.prediction.coordinate, (0.105, 0.10)), 0.01)

    def test_discrete_density_mode_returns_input_candidate(self):
        predictions = [
            Prediction("click", 0.10, 0.10, source="a"),
            Prediction("click", 0.11, 0.10, source="b"),
            Prediction("click", 0.90, 0.90, source="c"),
        ]
        coordinate = coordinate_density_medoid("androidcontrol", predictions, "click")
        self.assertIn(coordinate, [prediction.coordinate for prediction in predictions])
        self.assertNotEqual(coordinate, (0.105, 0.10))

    def test_mind2web_inference_kernel_has_no_bbox_argument(self):
        signature = inspect.signature(kernels.mind2web_coord_inference)
        self.assertEqual(list(signature.parameters), ["left", "right"])
        self.assertNotIn("gt_analysis", inspect.getsource(pka))

    def test_out_of_domain_points_are_preserved_without_clipping(self):
        self.assertEqual(kernels.mind2web_coord_inference((134.0, 180.0), (134.0, 180.0)), 1.0)
        self.assertEqual(kernels.mind2web_coord_inference((134.0, 180.0), (0.5, 0.5)), 0.0)

    def test_sequential_density_keeps_plurality_type(self):
        predictions = [
            Prediction("CLICK", 0.10, 0.10, source="a"),
            Prediction("CLICK", 0.11, 0.10, source="b"),
            Prediction("TYPE", 0.10, 0.10, "query", source="c"),
        ]
        result = plurality_then_density("mind2web", predictions, ["a", "b", "c"])
        self.assertEqual(result.prediction.action, "CLICK")


if __name__ == "__main__":
    unittest.main()