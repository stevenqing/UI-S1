import ast
import sys
import tempfile
import unittest
from pathlib import Path

import torch


sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_real_data_smoke import (
    build_result, load_config, metric_free_smoke_loss, validate_authorization,
)
from test_model import make_batch
from trivus_model import TriVUSSetRanker, trivus_loss


def prohibited_training_tokens(source):
    tree = ast.parse(source)
    names = {node.id.lower() for node in ast.walk(tree) if isinstance(node, ast.Name)}
    attributes = {node.attr.lower() for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    imports = set()
    strings = {
        node.value.lower() for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.lower() for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.add(str(node.module or "").lower())
            imports.update(alias.name.lower() for alias in node.names)
    forbidden = {
        "optimizer", "optim", "torch.optim", "adam", "adamw", "backward",
        "step", "zerograd", "zero_grad", "requires_grad", "requires_grad_",
        "parameters", "named_parameters", "state_dict", "load_state_dict",
        "data", "copy_", "add_", "sub_", "mul_", "div_", "set_", "setattr",
        "delattr", "register_parameter", "register_buffer", "add_module",
        "register_module", "set_submodule", "apply", "train", "to_empty",
        "autograd", "grad",
    }
    observed = names | attributes | imports | strings
    violations = set(forbidden & observed)

    def rooted_at_model(node):
        is_state_path = isinstance(node, (ast.Attribute, ast.Subscript))
        while isinstance(node, (ast.Attribute, ast.Subscript)):
            node = node.value
        return is_state_path and isinstance(node, ast.Name) and node.id == "model"

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr.endswith("_") and not node.func.attr.startswith("__"):
                violations.add(node.func.attr)
        if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(rooted_at_model(target) for target in targets):
                violations.add("model_state_assignment")
        if isinstance(node, ast.Delete) and any(rooted_at_model(target) for target in node.targets):
            violations.add("model_state_delete")
    return violations


class RealDataSmokeTest(unittest.TestCase):
    def test_frozen_metric_free_smoke_contract(self):
        config = load_config()
        self.assertEqual(config["phase"]["model_training_folds"], [3, 4])
        self.assertEqual(config["phase"]["checkpoint_fold"], 2)
        self.assertEqual(config["phase"]["holdout_fold"], 1)
        self.assertEqual(config["maximum_forward_rows"], 64)

    def test_source_has_no_optimizer_backward_or_parameter_step(self):
        path = Path(__file__).resolve().parent / "run_real_data_smoke.py"
        self.assertFalse(prohibited_training_tokens(path.read_text()))

    def test_static_gate_rejects_optimizer_gradient_and_parameter_mutation(self):
        snippets = (
            "import torch.optim as x\n",
            "from torch import optim\n",
            "model.backward()\n",
            "getattr(model, 'backward')()\n",
            "for value in model.parameters(): value.data.add_(1)\n",
            "model.load_state_dict({})\n",
            "value.requires_grad_(True)\n",
            "model.keep_delta = value\n",
            "del model.keep_delta\n",
            "value.masked_fill_(mask, 0)\n",
            "model.register_buffer('hidden', value)\n",
            "setattr(model, 'keep_delta', value)\n",
            "torch.autograd.grad(loss, inputs)\n",
        )
        for source in snippets:
            with self.subTest(source=source):
                self.assertTrue(prohibited_training_tokens(source))

    def test_result_schema_contains_no_numeric_private_metric(self):
        config = load_config()
        run_dir = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory(prefix=".smoke-result-test-", dir=run_dir) as directory:
            receipt = Path(directory) / "receipt.json"
            receipt.write_text("{}\n")
            result = build_result(
                config,
                {"implementation_commit": "a" * 40, "authorization_nonce": "b" * 64},
                "c" * 40,
                receipt,
                {"authorization_sha256": "d" * 64},
                [{"path": "fold.jsonl", "sha256": "e" * 64}],
            )
        self.assertEqual(set(result), set(config["result_allowed_fields"]))
        self.assertFalse(result["optimizer_constructed"])
        self.assertFalse(result["backward_called"])
        self.assertFalse(result["performance_metric_computed"])
        self.assertFalse(result["training_started"])
        forbidden = {"loss", "accuracy", "success", "active", "target", "auroc"}
        self.assertFalse(forbidden & set(result))

    def test_metric_free_loss_matches_frozen_loss_tensor(self):
        batch = make_batch(counts=(3, 12), input_dim=17)
        model = TriVUSSetRanker(17, dropout=0).eval()
        with torch.no_grad():
            expected, _ = trivus_loss(model, batch)
            observed = metric_free_smoke_loss(model, batch)
        self.assertTrue(torch.allclose(observed, expected, atol=1e-7, rtol=0))

    def test_execution_fails_closed_without_committed_authorization(self):
        config = load_config()
        original = config["authorization"]
        config["authorization"] = "runs/trivus/2026-08-12/missing-smoke-authorization.json"
        try:
            with self.assertRaises(PermissionError):
                validate_authorization(config)
        finally:
            config["authorization"] = original


if __name__ == "__main__":
    unittest.main()