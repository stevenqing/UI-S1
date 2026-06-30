import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from gui360_long_horizon.analysis.guards import FormatMismatchError, RecipeDiffError, assert_format_match, assert_no_v1_utilization_claim, assert_recipe_diff_only
from gui360_long_horizon.harness.rollout import UnwiredHarness, sharegpt_to_openai_messages
from train_GUI_360.build_history_arms import _assert_image_count, _compact_history_arm_human, _strip_inline_history, build_gt_history_example, build_own_history_example


def _png_bytes(color=(255, 0, 0)):
    import io

    buf = io.BytesIO()
    Image.new("RGB", (2, 2), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _row():
    steps = [
        {
            "step_idx": 0,
            "action": {"action": "click", "coordinate": [10, 20]},
            "screenshot": "/tmp/train/image/excel/in_app/success/e/action_step1.png",
            "bbox": [0, 0, 20, 30],
            "conversation_human": "<image>\nDo task step 1",
            "conversation_gpt": '<tool_call>{"function":"click","args":{"coordinate":[10,20]},"status":"CONTINUE"}</tool_call>',
        },
        {
            "step_idx": 1,
            "action": {"action": "click", "coordinate": [30, 40]},
            "screenshot": "/tmp/train/image/excel/in_app/success/e/action_step2.png",
            "bbox": [20, 30, 40, 50],
            "conversation_human": "<image>\nDo task step 2",
            "conversation_gpt": '<tool_call>{"function":"click","args":{"coordinate":[30,40]},"status":"CONTINUE"}</tool_call>',
        },
    ]
    return {"episode_id": 1, "steps": json.dumps(steps), "screenshots": [{"bytes": _png_bytes()}, {"bytes": _png_bytes((0, 255, 0))}]}


class HistoryArmTests(unittest.TestCase):
    def test_image_count_invariant(self):
        conversations = [{"from": "human", "value": "<image> a"}, {"from": "gpt", "value": "b"}]
        _assert_image_count(conversations, ["one.png"])
        with self.assertRaises(ValueError):
            _assert_image_count(conversations, [])

    def test_gt_history_builds_one_multiturn_episode(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            example = build_gt_history_example(_row(), split="train", image_root=Path(tmpdir), require_images=True)
        self.assertIsNotNone(example)
        self.assertEqual(len(example["images"]), 2)
        self.assertEqual(len(example["conversations"]), 4)
        self.assertEqual(example["conversations"][0]["from"], "human")
        self.assertEqual(example["conversations"][1]["from"], "gpt")

    def test_inline_history_is_removed_for_history_arms(self):
        human = "<image>\nInstruction\n\nThe history of actions are: Step 1: old action\nStep 2: older\n\nThe actions supported are:\n<actions>"
        stripped = _strip_inline_history(human)
        self.assertIn("The history of actions are: None", stripped)
        self.assertNotIn("old action", stripped)
        self.assertIn("The actions supported are:", stripped)

    def test_history_arm_human_prompt_is_compact(self):
        human = "<image>\nThe instruction is:\nCreate a report.\n\nThe history of actions are: Step 1: noisy\n\nThe actions supported are:\n" + "very long schema\n" * 100
        compact = _compact_history_arm_human(human)
        self.assertIn("Instruction: Create a report.", compact)
        self.assertIn("previous turns", compact)
        self.assertNotIn("noisy", compact)
        self.assertLess(len(compact), 700)

    def test_own_history_requires_wired_harness(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(RuntimeError):
                build_own_history_example(_row(), split="train", image_root=Path(tmpdir), require_images=True, harness=UnwiredHarness(), patch_budget=1)

    def test_sharegpt_to_openai_messages_carries_history_images(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "screen.png"
            image_path.write_bytes(_png_bytes())
            messages = sharegpt_to_openai_messages(
                [
                    {"from": "human", "value": "<image>\nFirst"},
                    {"from": "gpt", "value": "first action"},
                    {"from": "human", "value": "<image>\nSecond"},
                ],
                [str(image_path), str(image_path)],
            )
        self.assertEqual([message["role"] for message in messages], ["user", "assistant", "user"])
        self.assertEqual(messages[0]["content"][0]["type"], "image_url")
        self.assertEqual(messages[2]["content"][0]["type"], "image_url")
        with self.assertRaises(ValueError):
            sharegpt_to_openai_messages([{"from": "human", "value": "<image>"}], [])

    def test_format_and_v1_guards(self):
        assert_format_match("S", "none")
        assert_format_match("gt_history", "gt_history")
        with self.assertRaises(FormatMismatchError):
            assert_format_match("gt_history", "none")
        with self.assertRaises(RuntimeError):
            assert_no_v1_utilization_claim("V1")

    def test_recipe_diff_guard(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ref = root / "s.yaml"
            good = root / "g.yaml"
            bad = root / "bad.yaml"
            ref.write_text("dataset: s\noutput_dir: out_s\nrun_name: s\nlearning_rate: 1e-5\n", encoding="utf-8")
            good.write_text("dataset: g\noutput_dir: out_g\nrun_name: g\nlearning_rate: 1e-5\n", encoding="utf-8")
            bad.write_text("dataset: g\noutput_dir: out_g\nrun_name: g\nlearning_rate: 2e-5\n", encoding="utf-8")
            report = assert_recipe_diff_only(ref, [good])
            self.assertIn(str(good), report)
            with self.assertRaises(RecipeDiffError):
                assert_recipe_diff_only(ref, [bad])


if __name__ == "__main__":
    unittest.main()
