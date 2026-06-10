#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm


HF_SPLIT_REPO = "hflqf88888/GUIOdyssey"
HF_IMAGE_REPO = "OpenGVLab/GUI-Odyssey"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a selective GUI-Odyssey subset for CHORUS preflight/eval")
    parser.add_argument("--data-dir", default="datasets/GUI-Odyssey", help="Local GUI-Odyssey directory")
    parser.add_argument("--split", default="random_split", choices=["random_split", "app_split", "device_split", "task_split"])
    parser.add_argument("--subset", default="test", choices=["train", "test"])
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    split_path = hf_hub_download(
        HF_SPLIT_REPO,
        f"splits/{args.split}.json",
        repo_type="dataset",
        local_dir=str(data_dir),
    )
    with open(split_path, "r", encoding="utf-8") as file:
        split_data = json.load(file)
    episode_ids = [episode.replace(".json", "") for episode in split_data[args.subset]]
    print(f"Split {args.split}/{args.subset}: {len(episode_ids)} episodes")

    annotation_paths = download_annotations(data_dir, episode_ids, args.workers)
    required_screenshots = collect_required_screenshots(annotation_paths)
    print(f"Required screenshots: {len(required_screenshots)}")

    image_index = build_image_index(required_screenshots)
    missing_from_index = sorted(required_screenshots - set(image_index))
    if missing_from_index:
        print(f"Missing screenshots in {HF_IMAGE_REPO}: {len(missing_from_index)}")
        print("Examples:", missing_from_index[:20])
        return 2

    download_screenshots(data_dir, image_index, args.workers)
    print("Done")
    return 0


def download_annotations(data_dir: Path, episode_ids: Sequence[str], workers: int) -> List[Path]:
    paths: List[Path] = []
    existing = []
    pending = []
    for episode_id in episode_ids:
        local_path = data_dir / "annotations" / f"{episode_id}.json"
        if local_path.exists():
            existing.append(local_path)
        else:
            pending.append(episode_id)
    paths.extend(existing)

    def fetch(episode_id: str) -> Path:
        path = hf_hub_download(
            HF_SPLIT_REPO,
            f"annotations/{episode_id}.json",
            repo_type="dataset",
            local_dir=str(data_dir),
        )
        return Path(path)

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch, episode_id) for episode_id in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc="annotations"):
                paths.append(future.result())
    print(f"Annotations: existing={len(existing)} downloaded={len(pending)}")
    return paths


def collect_required_screenshots(annotation_paths: Iterable[Path]) -> Set[str]:
    screenshots: Set[str] = set()
    for path in annotation_paths:
        with open(path, "r", encoding="utf-8") as file:
            annotation = json.load(file)
        for step in annotation.get("steps", []):
            screenshot = step.get("screenshot")
            if screenshot:
                screenshots.add(os.path.basename(screenshot))
    return screenshots


def build_image_index(required: Set[str]) -> Dict[str, str]:
    api = HfApi()
    info = api.dataset_info(HF_IMAGE_REPO)
    index: Dict[str, str] = {}
    remaining = set(required)
    for sibling in info.siblings:
        filename = sibling.rfilename
        if not filename.startswith("screenshots/"):
            continue
        basename = os.path.basename(filename)
        if basename in remaining:
            index[basename] = filename
            remaining.remove(basename)
            if not remaining:
                break
    print(f"Indexed screenshots: {len(index)}")
    return index


def download_screenshots(data_dir: Path, image_index: Dict[str, str], workers: int) -> None:
    image_cache_dir = data_dir / "hf_images"
    flat_dir = data_dir / "data" / "screenshots"
    flat_dir.mkdir(parents=True, exist_ok=True)

    pending: List[Tuple[str, str]] = []
    linked = 0
    for basename, remote_path in image_index.items():
        flat_path = flat_dir / basename
        if flat_path.exists():
            linked += 1
            continue
        pending.append((basename, remote_path))

    def fetch_and_link(item: Tuple[str, str]) -> str:
        basename, remote_path = item
        local_path = Path(
            hf_hub_download(
                HF_IMAGE_REPO,
                remote_path,
                repo_type="dataset",
                local_dir=str(image_cache_dir),
            )
        )
        flat_path = flat_dir / basename
        if not flat_path.exists():
            relative_target = os.path.relpath(local_path, flat_path.parent)
            os.symlink(relative_target, flat_path)
        return basename

    if pending:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(fetch_and_link, item) for item in pending]
            for future in tqdm(as_completed(futures), total=len(futures), desc="screenshots"):
                future.result()
    print(f"Screenshots: existing={linked} downloaded_or_linked={len(pending)}")


if __name__ == "__main__":
    raise SystemExit(main())
