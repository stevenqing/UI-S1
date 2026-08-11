import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from router_model import AcquisitionRouter, RouterBatch, permute_router_batch, router_loss
from router_data import fit_source_statistics, source_reliability_values
from router_train import load_test_after_pretest


class RouterTest(unittest.TestCase):
    def test_source_reliability_is_leave_one_and_benchmark_scoped(self):
        metadata = {
            ("mind2web", "row-a"): {"benchmark": "mind2web", "row_id": "row-a", "fold": 0, "sources": ["s"] * 6},
            ("screenspot_pro", "row-b"): {"benchmark": "screenspot_pro", "row_id": "row-b", "fold": 0, "sources": ["s"] * 6},
        }
        labels = {
            "mind2web/C_uni/row-a": {"candidate_success": [True, False, False, False, False, False]},
            "screenspot_pro/C_uni/row-b": {"candidate_success": [True] * 6},
        }
        stats = fit_source_statistics(metadata, labels, [0])
        values = source_reliability_values("mind2web", "row-a", metadata, labels, stats, leave_one=True)
        self.assertAlmostEqual(values[0], 1 / 7)
        self.assertAlmostEqual(values[1], 2 / 7)
        fixed = source_reliability_values("screenspot_pro", "row-b", metadata, labels, stats, leave_one=False)
        self.assertTrue(all(abs(value - 7 / 8) < 1e-12 for value in fixed))

    def test_permutation_invariant(self):
        torch.manual_seed(11); model=AcquisitionRouter(9,dropout=0).eval(); features=torch.randn(4,6,9); permutations=torch.stack([torch.randperm(6) for _ in range(4)])
        batch=RouterBatch(features,torch.zeros(4,4),torch.zeros(4,4),torch.zeros(4),torch.ones(4)); changed=permute_router_batch(batch,permutations)
        with torch.no_grad(): self.assertTrue(torch.allclose(model(features),model(changed.features),atol=1e-5,rtol=1e-5))

    def test_loss_learns_positive_arm(self):
        torch.manual_seed(12); model=AcquisitionRouter(8,width=32,layers=1,dropout=0); features=torch.randn(16,6,8); targets=torch.zeros(16,4); targets[:,2]=1; batch=RouterBatch(features,targets,targets.clone(),torch.ones(16),torch.ones(16)); optimizer=torch.optim.AdamW(model.parameters(),lr=3e-3)
        before=model(features).softmax(-1)[:,2].mean().item()
        for _ in range(30): optimizer.zero_grad(); loss,_=router_loss(model,batch); loss.backward(); optimizer.step()
        after=model(features).softmax(-1)[:,2].mean().item(); self.assertGreater(after,before+.4)

    def test_outer_labels_sealed_until_pretest(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); labels=root/'private_labels_fold-2.jsonl'; labels.write_text(json.dumps({'sample_key':'key','candidate_success':[False]*12})+'\n'); pre=root/'outer-2.pretest.json'
            with self.assertRaises(PermissionError): load_test_after_pretest(2,pre)


if __name__=='__main__': unittest.main()
