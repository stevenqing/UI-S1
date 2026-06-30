#!/usr/bin/env python3
"""Build an OCR cache for local ``datasets/gui360-balanced`` parquet files.

The dependency diagnostic consumes this cache as an offline referee input. The
cache is never placed into model prompts.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


OCR_ENGINE = None

for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_thread_env, "1")


@dataclass(frozen=True)
class OcrTarget:
    step_uid: str
    split: str
    episode_id: Any
    step_id: int
    screenshot: str
    image_bytes: bytes


def _step_uid(split: str, episode_id: Any, step_id: int) -> str:
    return f"gui360-balanced:{split}:{episode_id}:step{step_id}"


def _image_bytes(cell: Any) -> Optional[bytes]:
    if not isinstance(cell, dict):
        return None
    data = cell.get("bytes")
    return bytes(data) if isinstance(data, (bytes, bytearray)) else None


def _screenshot_key(step: Dict[str, Any], fallback: str) -> str:
    return str(step.get("screenshot") or fallback)


def iter_targets(data_dir: str | Path, split: str, *, limit: int = 0) -> Iterable[OcrTarget]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("pyarrow is required; run this script in the uv diagnostic environment") from exc

    root = Path(data_dir)
    files = sorted(root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split}-*.parquet files under {root}")
    emitted = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=128, columns=["episode_id", "steps", "screenshots"]):
            for row in batch.to_pylist():
                episode_id = row.get("episode_id")
                try:
                    steps = json.loads(row.get("steps") or "[]")
                except json.JSONDecodeError:
                    continue
                screenshots = row.get("screenshots") or []
                if not isinstance(steps, list) or not isinstance(screenshots, list):
                    continue
                for step_pos, step in enumerate(steps):
                    if not isinstance(step, dict):
                        continue
                    step_id = int(step.get("step_idx", step.get("step_id", step_pos)) or step_pos)
                    cell = screenshots[step_pos] if step_pos < len(screenshots) else None
                    payload = _image_bytes(cell)
                    if not payload:
                        continue
                    fallback = f"{split}/episode_{episode_id}/step_{step_id:04d}.png"
                    yield OcrTarget(
                        step_uid=_step_uid(split, episode_id, step_id),
                        split=split,
                        episode_id=episode_id,
                        step_id=step_id,
                        screenshot=_screenshot_key(step, fallback),
                        image_bytes=payload,
                    )
                    emitted += 1
                    if limit > 0 and emitted >= limit:
                        return


def init_ocr() -> None:
    global OCR_ENGINE
    if OCR_ENGINE is None:
        from rapidocr_onnxruntime import RapidOCR

        OCR_ENGINE = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)


def _image_array(image_bytes: bytes) -> Any:
    import numpy as np
    from PIL import Image

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.asarray(image)


def run_ocr(target: OcrTarget) -> Dict[str, Any]:
    init_ocr()
    start = time.time()
    try:
        result, _ = OCR_ENGINE(_image_array(target.image_bytes))
        texts = []
        for item in result or []:
            box, text, score = item
            if text:
                texts.append({"text": str(text), "score": float(score), "box": box})
        return {
            "step_uid": target.step_uid,
            "screenshot": target.screenshot,
            "image_rel_path": target.screenshot,
            "split": target.split,
            "episode_id": target.episode_id,
            "step_id": target.step_id,
            "ok": True,
            "error": "",
            "elapsed": time.time() - start,
            "texts": texts,
            "ocr_text": " ".join(item["text"] for item in texts),
        }
    except Exception as exc:
        return {
            "step_uid": target.step_uid,
            "screenshot": target.screenshot,
            "image_rel_path": target.screenshot,
            "split": target.split,
            "episode_id": target.episode_id,
            "step_id": target.step_id,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed": time.time() - start,
            "texts": [],
            "ocr_text": "",
        }


def _done_step_uids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("step_uid"):
                done.add(str(row["step_uid"]))
    return done


def dependency_target_step_uids(data_dir: str | Path, splits: Sequence[str]) -> Set[str]:
    from gui360_long_horizon.analysis.dependency_diag import load_balanced_parquet
    from gui360_long_horizon.data.carried_value import extract_candidates, step_uid

    targets: Set[str] = set()
    for split in splits:
        episodes = load_balanced_parquet(data_dir, split)
        extraction = extract_candidates(episodes)
        for candidate in extraction.candidates:
            if candidate.layer0_bucket:
                continue
            for step_index in range(candidate.produce_index + 1, candidate.consume_index + 1):
                trajectory = next((episode for episode in episodes if episode and getattr(episode[0], "episode_id", None) == candidate.episode_id), None)
                if trajectory is not None and 0 <= step_index < len(trajectory):
                    targets.add(step_uid(trajectory[step_index], step_index))
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCR cache for gui360-balanced parquet screenshots")
    parser.add_argument("--data-dir", default="datasets/gui360-balanced/data")
    parser.add_argument("--split", action="append", default=[], help="Split to process; repeat for train/test. Defaults to train.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--flush-every", type=int, default=25)
    parser.add_argument("--dependency-targets-only", action="store_true", help="OCR only screens needed by dependency candidate windows")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    splits = args.split or ["train"]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done = _done_step_uids(output)
    target_uids = dependency_target_step_uids(args.data_dir, splits) if args.dependency_targets_only else None
    targets: List[OcrTarget] = []
    for split in splits:
        for target in iter_targets(args.data_dir, split, limit=args.limit):
            if target_uids is not None and target.step_uid not in target_uids:
                continue
            if target.step_uid not in done:
                targets.append(target)
    print(json.dumps({"output": str(output), "splits": splits, "done": len(done), "dependency_targets": len(target_uids) if target_uids is not None else None, "pending": len(targets)}, indent=2, ensure_ascii=False))
    if not targets:
        return 0
    with output.open("a", encoding="utf-8") as handle:
        if args.workers <= 1:
            for idx, target in enumerate(targets, 1):
                row = run_ocr(target)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                if idx % max(1, args.flush_every) == 0:
                    handle.flush()
                    print(f"ocr {idx}/{len(targets)}")
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = [executor.submit(run_ocr, target) for target in targets]
                for idx, future in enumerate(as_completed(futures), 1):
                    row = future.result()
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    if idx % max(1, args.flush_every) == 0:
                        handle.flush()
                        print(f"ocr {idx}/{len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())