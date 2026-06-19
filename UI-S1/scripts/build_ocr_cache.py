#!/usr/bin/env python3
"""Build a resumable OCR cache for screenshots referenced by CMU data."""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Iterable


OCR_ENGINE = None
MAX_SIDE = 0


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_screenshots(paths: list[Path], labels: set[str], neutral_sample: int, seed: int) -> list[str]:
    selected = set()
    neutral = []
    for path in paths:
        for row in iter_jsonl(path):
            label = row.get("utility_label", "")
            screenshot = row.get("metadata", {}).get("screenshot")
            if not screenshot:
                continue
            if label in labels:
                selected.add(screenshot)
            elif label == "neutral" and neutral_sample > 0:
                neutral.append(screenshot)
    if neutral_sample > 0:
        rng = random.Random(seed)
        neutral_unique = sorted(set(neutral))
        rng.shuffle(neutral_unique)
        selected.update(neutral_unique[:neutral_sample])
    return sorted(selected)


def init_worker(max_side: int = 0) -> None:
    global OCR_ENGINE
    global MAX_SIDE
    MAX_SIDE = max_side
    from rapidocr_onnxruntime import RapidOCR
    OCR_ENGINE = RapidOCR()


def load_image(path: Path) -> Any:
    if MAX_SIDE <= 0:
        return str(path)
    import cv2
    image = cv2.imread(str(path))
    if image is None:
        return str(path)
    height, width = image.shape[:2]
    max_dim = max(height, width)
    if max_dim <= MAX_SIDE:
        return image
    scale = MAX_SIDE / max_dim
    new_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)


def run_ocr(path_text: str) -> dict[str, Any]:
    global OCR_ENGINE
    if OCR_ENGINE is None:
        init_worker(MAX_SIDE)
    path = Path(path_text)
    if not path.exists():
        return {"screenshot": path_text, "ok": False, "error": "missing file", "texts": [], "ocr_text": ""}
    try:
        image_input = load_image(path)
        result, elapsed = OCR_ENGINE(image_input)
        texts = []
        for item in result or []:
            box, text, score = item
            if text:
                texts.append({"text": str(text), "score": float(score), "box": box})
        ocr_text = " ".join(item["text"] for item in texts)
        return {"screenshot": path_text, "ok": True, "error": "", "max_side": MAX_SIDE, "elapsed": elapsed, "texts": texts, "ocr_text": ocr_text}
    except Exception as exc:
        return {"screenshot": path_text, "ok": False, "error": f"{type(exc).__name__}: {exc}", "texts": [], "ocr_text": ""}


def load_done(path: Path) -> set[str]:
    done = set()
    if not path.exists():
        return done
    for row in iter_jsonl(path):
        screenshot = row.get("screenshot")
        if screenshot:
            done.add(str(screenshot))
    return done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build OCR cache for screenshots")
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--labels", nargs="+", default=["positive", "negative", "nonspecific_positive", "summary_insufficient", "unresolved"])
    parser.add_argument("--neutral-sample", type=int, default=1000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--max-side", type=int, default=960)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    screenshots = collect_screenshots([Path(path) for path in args.inputs], set(args.labels), args.neutral_sample, args.seed)
    if args.limit > 0:
        screenshots = screenshots[:args.limit]
    done = load_done(output)
    pending = [path for path in screenshots if path not in done]
    print(json.dumps({"target": len(screenshots), "done": len(done), "pending": len(pending), "output": str(output)}, indent=2))
    if not pending:
        return
    with output.open("a", encoding="utf-8") as handle:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=init_worker, initargs=(args.max_side,)) as executor:
            futures = {executor.submit(run_ocr, path): path for path in pending}
            completed = 0
            for future in as_completed(futures):
                row = future.result()
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                completed += 1
                if completed % 50 == 0:
                    handle.flush()
                    print(f"ocr {completed}/{len(pending)}")


if __name__ == "__main__":
    main()