import json
import unittest
from pathlib import Path

import yaml

from sourcebias_common import b3_select_index, graph_component


RUN_DIR = Path(__file__).resolve().parent


class SourceBiasContractTest(unittest.TestCase):
    def test_b3_coverage_tie_break(self):
        candidates=[{"point":[0,0],"coverage":0},{"point":[1,1],"coverage":3},{"point":[100,100],"coverage":99}]
        selected,group=b3_select_index(candidates)
        self.assertEqual(group,(0,1)); self.assertEqual(selected,1)

    def test_graph_nearest_source_attribution(self):
        candidates=[{"point":[0,0]},{"point":[2,0]},{"point":[100,100]}]
        selected,component,point=graph_component(candidates)
        self.assertEqual(component,(0,1)); self.assertEqual(point,[1.0,0.0]); self.assertEqual(selected,0)

    def test_frozen_variant_grid(self):
        config=yaml.safe_load((RUN_DIR/"configs/b2_variants.yaml").read_text())
        self.assertEqual(len(config["variant_order"]),21); self.assertEqual(len(set(config["variant_order"])),21)

    def test_complete_results_and_gates(self):
        b1=json.loads((RUN_DIR/"results/b1_source_bias.json").read_text()); b2=json.loads((RUN_DIR/"results/b2_lineage_normalized.json").read_text()); b4=json.loads((RUN_DIR/"results/b4_attribution.json").read_text())
        self.assertTrue(b1["gate"]["B1_pass"]); self.assertEqual(b2["variant_count"],21); self.assertFalse(b2["B2_primary_success"]); self.assertTrue(b2["B_K4"]); self.assertEqual(b2["B3x_action"],"CANCEL"); self.assertEqual(b4["proposal_source_attribution"]["interpretation"],"heterogeneous_pool_aggregation_effect")
        for scale in ("7B","72B"):
            self.assertEqual(len(b2["reports"][scale]["outer_selections"]),5)
            self.assertEqual(len(b2["reports"][scale]["descriptive_crossfit_grid"]),21)
            self.assertEqual(len(b2["reports"][scale]["outputs"]["nested_LN"]),1581)


if __name__=="__main__": unittest.main()
