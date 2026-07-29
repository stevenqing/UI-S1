import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_METADATA_SHA256 = "6e86b61ab6b8c657cabadc73de9df1f844dc39e4904228c8b2b5a18b68640d2d"
EXPECTED_ROWS = 2080
EXPECTED_EPISODES = 252


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=Path("data/evaluation_data"))
    args = parser.parse_args()

    metadata_path = args.metadata.resolve()
    annotations_path = args.annotations.resolve()
    images_dir = args.images.resolve()
    if sha256(metadata_path) != EXPECTED_METADATA_SHA256:
        raise ValueError("TongUI thought metadata hash mismatch")

    rows = json.loads(metadata_path.read_text())
    annotations = json.loads(annotations_path.read_text())
    expected = {}
    for episode in annotations:
        for step, step_repr in zip(episode["actions"], episode["action_reprs"]):
            filename = f'{episode["annotation_id"]}-{step["action_uid"]}.jpg'
            if (images_dir / filename).exists():
                expected[(episode["annotation_id"], step["action_uid"])] = (
                    episode,
                    step,
                    step_repr,
                    filename,
                )

    actual = {(row["annot_id"], row["action_uid"]): row for row in rows}
    if len(rows) != len(actual) or len(actual) != EXPECTED_ROWS:
        raise ValueError("TongUI metadata contains missing or duplicate identities")
    if set(actual) != set(expected):
        raise ValueError("TongUI metadata identities do not match scoreable Mind2Web actions")

    for identity, row in actual.items():
        episode, step, step_repr, filename = expected[identity]
        if (
            row["img_url"] != filename
            or row["step"]["bbox"] != step["bbox"]
            or row["step"]["operation"] != step["operation"]
            or row["step_repr"] != step_repr
            or row["task"] != episode["confirmed_task"]
        ):
            raise ValueError(f"ground-truth mismatch for identity {identity}")
    if len({row["annot_id"] for row in rows}) != EXPECTED_EPISODES:
        raise ValueError("unexpected episode count")

    mind2web_dir = args.output_root.resolve() / "Mind2Web"
    output_metadata_dir = mind2web_dir / "metadata"
    output_metadata_dir.mkdir(parents=True, exist_ok=True)
    output_metadata = output_metadata_dir / "hf_test_task_with_thoughts.json"
    if output_metadata.exists() or output_metadata.is_symlink():
        if output_metadata.resolve() != metadata_path:
            raise ValueError("existing metadata link points to a different file")
    else:
        output_metadata.symlink_to(metadata_path)

    image_link = mind2web_dir / "ming2web_images"
    if image_link.exists() or image_link.is_symlink():
        if image_link.resolve() != images_dir:
            raise ValueError("existing image link points to a different directory")
    else:
        image_link.symlink_to(images_dir, target_is_directory=True)

    manifest = {
        "benchmark_repository": "Bofeee5675/GUI-Net-Benchmark",
        "benchmark_revision": "b212dcd803bd1be8318b9cfbe4912f385a748ff7",
        "metadata": str(metadata_path),
        "metadata_bytes": metadata_path.stat().st_size,
        "metadata_sha256": sha256(metadata_path),
        "annotations": str(annotations_path),
        "annotations_sha256": sha256(annotations_path),
        "images": str(images_dir),
        "rows": len(rows),
        "episodes": len({row["annot_id"] for row in rows}),
        "identities": len(actual),
        "history_entries": sum(len(row["step_history"]) for row in rows),
        "history_entries_with_thoughts": sum(
            "thoughts" in history
            for row in rows
            for history in row["step_history"]
        ),
    }
    manifest_path = args.output_root.resolve().parent / "data_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()