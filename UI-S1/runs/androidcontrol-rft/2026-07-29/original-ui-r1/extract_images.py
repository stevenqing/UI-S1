import argparse
import gzip
import hashlib
import json
import struct
from pathlib import Path

from tfrecord import example_pb2


def iter_tfrecord(path: Path):
    with gzip.open(path, "rb") as handle:
        while True:
            length_bytes = handle.read(8)
            if not length_bytes:
                return
            if len(length_bytes) != 8:
                raise ValueError(f"truncated record length in {path}")
            length = struct.unpack("<Q", length_bytes)[0]
            if len(handle.read(4)) != 4:
                raise ValueError(f"truncated length CRC in {path}")
            payload = handle.read(length)
            if len(payload) != length:
                raise ValueError(f"truncated record payload in {path}")
            if len(handle.read(4)) != 4:
                raise ValueError(f"truncated data CRC in {path}")
            yield payload


def feature_values(example, name: str, kind: str):
    feature = example.features.feature[name]
    return list(getattr(feature, f"{kind}_list").value)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--prepared-images", type=Path, required=True)
    parser.add_argument("--tfrecord-dir", type=Path, required=True)
    parser.add_argument("--output-images", type=Path, required=True)
    parser.add_argument("--output-manifest", type=Path, required=True)
    args = parser.parse_args()

    source = json.loads(args.source_json.read_text())
    expected = {
        row["image"].replace("-screenshot_", "_screenshot_"): row
        for row in source
    }
    if len(source) != 7868 or len(expected) != 7868:
        raise ValueError("original UI-R1 source must contain 7868 unique rows")
    present = {name for name in expected if (args.prepared_images / name).is_file()}
    missing = set(expected) - present
    needed_episodes = {int(name.split("_")[1]) for name in missing}
    args.output_images.mkdir(parents=True, exist_ok=True)
    recovered = {}
    scanned_episodes = 0

    for tfrecord_path in sorted(args.tfrecord_dir.glob("android_control-*-of-00020")):
        for payload in iter_tfrecord(tfrecord_path):
            example = example_pb2.Example()
            example.ParseFromString(payload)
            episode_id = feature_values(example, "episode_id", "int64")[0]
            scanned_episodes += 1
            if episode_id not in needed_episodes:
                continue
            screenshots = feature_values(example, "screenshots", "bytes")
            for step_id, image_bytes in enumerate(screenshots):
                name = f"episode_{episode_id}_screenshot_{step_id}.png"
                if name not in missing:
                    continue
                if not image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
                    raise ValueError(f"non-PNG screenshot in official record: {name}")
                output = args.output_images / name
                output.write_bytes(image_bytes)
                recovered[name] = {
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                    "bytes": len(image_bytes),
                    "source_shard": tfrecord_path.name,
                    "source_episode_id": episode_id,
                    "source_step_id": step_id,
                }
    unresolved = sorted(missing - set(recovered))
    if unresolved:
        raise ValueError(f"official TFRecords did not resolve {len(unresolved)} images: {unresolved[:20]}")
    result = {
        "status": "COMPLETE",
        "source_rows": len(source),
        "prepared_images_reused": len(present),
        "official_images_recovered": len(recovered),
        "total_image_coverage": len(present) + len(recovered),
        "scanned_episodes": scanned_episodes,
        "recovered": recovered,
    }
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output_manifest.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "recovered"}, indent=2))


if __name__ == "__main__":
    main()