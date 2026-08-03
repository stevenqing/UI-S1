import unittest

from h3_eval import evaluate_pool, pair_type


class H3MixedPoolTest(unittest.TestCase):
    def test_pair_types(self):
        a0 = {"model": "a", "view_index": 0}
        a1 = {"model": "a", "view_index": 1}
        b0 = {"model": "b", "view_index": 0}
        b1 = {"model": "b", "view_index": 1}
        self.assertEqual(pair_type(a0, a1), "same-model-diff-view")
        self.assertEqual(pair_type(a0, b0), "cross-lineage-same-view")
        self.assertEqual(pair_type(a0, b1), "cross-lineage-diff-view")

    def test_evaluate_pool_fixed_budget(self):
        rows = []
        for index in range(200):
            candidates = []
            for model_index, model in enumerate(("a", "b", "c")):
                for view_index in range(4):
                    near = index % 2 == 0 and model_index == 0
                    point = [10 + view_index, 10] if near else [500 + 10 * model_index, 500 + view_index]
                    candidates.append({
                        "model": model, "view_index": view_index, "point": point,
                        "region": [0, 0, 1000, 1000], "coverage": 4 - view_index,
                    })
            rows.append({
                "id": f"row-{index}", "application": f"app-{index % 10}",
                "target_bbox": [0, 0, 30, 30], "candidates": candidates,
            })
        result = evaluate_pool(rows)
        self.assertEqual(result["rows"], 200)
        self.assertEqual(sum(result["fold_rows"]), 200)
        self.assertEqual(set(result["accuracy"]), {"B3_mvp", "M1_ccm", "pass_at_12"})


if __name__ == "__main__":
    unittest.main()