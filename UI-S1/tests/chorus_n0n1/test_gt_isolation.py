from __future__ import annotations

import json
import random
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHORUS_ROOT = ROOT / "chorus-n0n1"
sys.path.insert(0, str(CHORUS_ROOT))

from src.readers.disagreement import reader_input_from_step_row  # noqa: E402
from src.readers.input import ReaderInput, build_reader_prompt  # noqa: E402


class GTIsolationTests(unittest.TestCase):
    def test_reader_input_rejects_gt_fields(self) -> None:
        with self.assertRaises(ValueError):
            ReaderInput({
                "goal": "open settings",
                "current_screenshot": "screen.png",
                "schema_payload": {},
                "episode_id": "ep",
                "step_idx": 0,
                "episode_len": 1,
                "gt_action": {"action": "click"},
            })

    def test_reader_prompts_do_not_contain_gt_action_text(self) -> None:
        rows = _load_step_rows()
        sample = random.Random(7).sample(rows, min(50, len(rows)))
        for row in sample:
            reader_input = reader_input_from_step_row(row)
            prompt = _normalize_text(build_reader_prompt(reader_input))
            gt_action_text = _normalize_text(json.dumps(row.get("gt_action") or {}, ensure_ascii=False, sort_keys=True))
            self.assertNotIn(gt_action_text, prompt)

    def test_disagreement_features_import_no_scoring_module(self) -> None:
        source = (CHORUS_ROOT / "src" / "readers" / "disagreement.py").read_text(encoding="utf-8")
        self.assertNotIn("bench.scoring", source)
        self.assertNotIn("src.bench.scoring", source)


def _load_step_rows() -> list[dict]:
    path = CHORUS_ROOT / "runs" / "n0n1_inputs" / "har_gui_odyssey_latest" / "har_gui_odyssey_steps.jsonl"
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


if __name__ == "__main__":
    unittest.main()