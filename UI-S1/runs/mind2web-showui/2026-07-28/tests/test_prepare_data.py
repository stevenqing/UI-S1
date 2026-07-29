import json
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from prepare_data import transform


class PrepareDataTest(unittest.TestCase):
    def test_history_entries_match_showui_dataset_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_dir = root / "images"
            image_dir.mkdir()
            annotation_id = "episode"
            actions = []
            action_reprs = []
            for index in range(2):
                action_uid = f"action-{index}"
                Image.new("RGB", (100, 50)).save(
                    image_dir / f"{annotation_id}-{action_uid}.jpg"
                )
                actions.append(
                    {
                        "action_uid": action_uid,
                        "operation": {"op": "CLICK", "value": ""},
                        "bbox": {"x": 1, "y": 2, "width": 3, "height": 4},
                    }
                )
                action_reprs.append(f"[button] value-{index} -> CLICK")
            episode = {
                "annotation_id": annotation_id,
                "confirmed_task": "test",
                "website": "site",
                "domain": "domain",
                "subdomain": "subdomain",
                "actions": actions,
                "action_reprs": action_reprs,
            }
            annotations = root / "annotations.json"
            annotations.write_text(json.dumps([episode]))

            import prepare_data

            old_counts = (
                prepare_data.EXPECTED_EPISODES,
                prepare_data.EXPECTED_ACTIONS,
                prepare_data.EXPECTED_ROWS,
            )
            prepare_data.EXPECTED_EPISODES = 1
            prepare_data.EXPECTED_ACTIONS = 2
            prepare_data.EXPECTED_ROWS = 2
            try:
                rows = transform(annotations, image_dir)
            finally:
                (
                    prepare_data.EXPECTED_EPISODES,
                    prepare_data.EXPECTED_ACTIONS,
                    prepare_data.EXPECTED_ROWS,
                ) = old_counts

            history = rows[1]["step_history"][0]
            self.assertEqual(history["step"], actions[0])
            self.assertEqual(history["step_repr"], action_reprs[0])
            self.assertEqual(history["img_url"], "episode-action-0.jpg")
            self.assertEqual(history["img_size"], [100, 50])
            self.assertEqual(history["task"], "test")

            repo_dir = Path(__file__).parents[1] / "repos" / "ShowUI"
            sys.path.insert(0, str(repo_dir))
            from data.dset_mind2web import get_answer

            self.assertEqual(
                get_answer(history, history["step"], history["step_repr"]),
                {
                    "action": "CLICK",
                    "value": "value-0",
                    "position": [0.03, 0.08],
                },
            )


if __name__ == "__main__":
    unittest.main()