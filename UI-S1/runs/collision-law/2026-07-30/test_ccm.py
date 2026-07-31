import unittest

from ccm import (
    CCMCalibration,
    RankLikelihoodRatio,
    candidate_class,
    collision_calibrated_mode,
    pair_type,
)
from pka import Prediction


class CollisionCalibratedModeTest(unittest.TestCase):
    def test_rank_likelihood_ratio_is_invariant_to_monotone_transform(self):
        observations = [
            (index / 128, index % 3 == 0)
            for index in range(128)
        ]
        original = RankLikelihoodRatio.fit(observations)
        transformed = RankLikelihoodRatio.fit([(value**3 + 7, label) for value, label in observations])
        self.assertEqual(original.log_ratios, transformed.log_ratios)
        for value, _ in observations:
            self.assertEqual(original.score(value), transformed.score(value**3 + 7))

    def test_candidate_classes_keep_parameterless_hedges_separate(self):
        self.assertEqual(candidate_class("androidcontrol", Prediction("wait")), "parameterless")
        self.assertEqual(candidate_class("androidcontrol", Prediction("click", 0.1, 0.2)), "coordinate-bearing")
        self.assertEqual(candidate_class("androidcontrol", Prediction("type", parameter="query")), "string-bearing")
        self.assertEqual(candidate_class("mind2web", Prediction("TYPE", 0.1, 0.2, "query")), "string-bearing")

    def test_pair_type_separates_views_families_and_lineages(self):
        full = Prediction("click", source="gui-r1-7b/full")
        view = Prediction("click", source="gui-r1-7b/v1")
        sibling = Prediction("click", source="gui-r1-3b/full")
        other = Prediction("click", source="ui-agile-7b/full")
        self.assertEqual(pair_type(full, view), "same-model-diff-view")
        self.assertEqual(pair_type(full, sibling), "same-family")
        self.assertEqual(pair_type(full, other), "cross-family")

    def test_map_prior_is_natural_fallback_when_lr_is_zero(self):
        table = RankLikelihoodRatio((), (0.0,) * 8, 32, 32)
        calibration = CCMCalibration(
            "androidcontrol",
            {"strong": 0.8, "weak": 0.2},
            table,
            {},
            {},
            {},
        )
        predictions = [
            Prediction("wait", source="strong"),
            Prediction("press_back", source="weak"),
        ]
        result = collision_calibrated_mode(calibration, predictions)
        self.assertEqual(result.prediction.source, "strong")


if __name__ == "__main__":
    unittest.main()