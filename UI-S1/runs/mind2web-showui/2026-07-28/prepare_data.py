import argparse
import hashlib
import json
import os
from pathlib import Path

from PIL import Image


EXPECTED_EPISODES = 252
EXPECTED_ACTIONS = 2094
EXPECTED_ROWS = 2080


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def transform(annotations_path: Path, images_dir: Path) -> list[dict]:
    episodes = json.loads(annotations_path.read_text())
    if len(episodes) != EXPECTED_EPISODES:
        raise ValueError(f"expected {EXPECTED_EPISODES} episodes, found {len(episodes)}")

    action_count = sum(len(episode["actions"]) for episode in episodes)
    if action_count != EXPECTED_ACTIONS:
        raise ValueError(f"expected {EXPECTED_ACTIONS} actions, found {action_count}")

    rows = []
    for episode in episodes:
        step_history = []
        repr_history = []
        for step, step_repr in zip(episode["actions"], episode["action_reprs"]):
            filename = f'{episode["annotation_id"]}-{step["action_uid"]}.jpg'
            image_path = images_dir / filename
            if not image_path.exists():
                continue
            if "bbox" not in step:
                raise ValueError(f"image exists for action without bbox: {filename}")
            with Image.open(image_path) as image:
                image_size = list(image.size)

            rows.append(
                {
                    "split": "test_task",
                    "id": f"mind2web_{len(rows)}",
                    "annot_id": episode["annotation_id"],
                    "action_uid": step["action_uid"],
                    "website": episode["website"],
                    "domain": episode["domain"],
                    "subdomain": episode["subdomain"],
                    "task": episode["confirmed_task"],
                    "img_url": filename,
                    "img_size": image_size,
                    "step_id": len(step_history),
                    "step": step,
                    "step_repr": step_repr,
                    "step_history": step_history.copy(),
                    "repr_history": repr_history.copy(),
                }
            )
            # ShowUI's dataset expects wrapped history entries here. Its released
            # converter appends the raw action and cannot read its own output.
            step_history.append(
                {
                    "step": step,
                    "step_repr": step_repr,
                    "img_url": filename,
                    "img_size": image_size,
                    "task": episode["confirmed_task"],
                }
            )
            repr_history.append(step_repr)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} scoreable rows, found {len(rows)}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data"))
    args = parser.parse_args()

    annotations_path = args.annotations.resolve()
    images_dir = args.images.resolve()
    mind2web_dir = args.output_root.resolve() / "Mind2Web"
    metadata_dir = mind2web_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    image_link = mind2web_dir / "images"
    if image_link.exists() or image_link.is_symlink():
        if image_link.resolve() != images_dir:
            raise ValueError(f"existing image link points to {image_link.resolve()}")
    else:
        image_link.symlink_to(images_dir, target_is_directory=True)

    rows = transform(annotations_path, images_dir)
    output_path = metadata_dir / "hf_test_task.json"
    output_path.write_text(json.dumps(rows, indent=2))
    manifest = {
        "annotations": str(annotations_path),
        "annotations_sha256": sha256(annotations_path),
        "images": str(images_dir),
        "episodes": EXPECTED_EPISODES,
        "actions": EXPECTED_ACTIONS,
        "rows": len(rows),
        "metadata": str(output_path),
        "metadata_sha256": sha256(output_path),
    }
    (args.output_root.resolve() / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
