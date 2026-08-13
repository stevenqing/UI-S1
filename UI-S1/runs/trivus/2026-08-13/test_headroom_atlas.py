import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parent))

from headroom_atlas import candidate_oracle, union_policy_outputs


class HeadroomAtlasTest(unittest.TestCase):
    def test_candidate_oracle_uses_any_candidate(self):
        self.assertEqual(
            candidate_oracle({
                "a": [False, True, False],
                "b": [False, False, False],
            }),
            {"a": True, "b": False},
        )

    def test_policy_union_uses_available_family_coverage(self):
        public = {
            "m": {"benchmark": "mind2web"},
            "a": {"benchmark": "androidcontrol"},
        }
        outputs = {
            policy: {"safe": {}, "direct": {}}
            for policy in (
                "JOINT3", "TARGET_ONLY", "JOINT2_NO_ANDROID", "NO_VISUAL",
                "RANDOM_ID_PLACEBO",
            )
        }
        for policy in outputs:
            outputs[policy]["safe"] = {"m": False, "a": False}
        outputs["JOINT2_NO_ANDROID"]["safe"].pop("a")
        outputs["TARGET_ONLY"]["safe"]["m"] = True
        outputs["JOINT3"]["safe"]["a"] = True
        values, contributors = union_policy_outputs(outputs, "safe", public)
        self.assertEqual(values, {"m": True, "a": True})
        self.assertEqual(contributors["m"], ("TARGET_ONLY",))
        self.assertEqual(contributors["a"], ("JOINT3",))

    def test_policy_union_rejects_missing_row(self):
        outputs = {
            policy: {"safe": {}}
            for policy in (
                "JOINT3", "TARGET_ONLY", "JOINT2_NO_ANDROID", "NO_VISUAL",
                "RANDOM_ID_PLACEBO",
            )
        }
        with self.assertRaisesRegex(ValueError, "No policy"):
            union_policy_outputs(outputs, "safe", {"missing": {}})


if __name__ == "__main__":
    unittest.main()