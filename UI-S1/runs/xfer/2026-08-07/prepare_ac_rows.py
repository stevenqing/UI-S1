import hashlib
import json
from pathlib import Path

from PIL import Image
from io import BytesIO

from ac_common import RUN_DIR, image_sha256, load_paired_sample, source_sha256


def main():
    rows = load_paired_sample()
    output_root = RUN_DIR / "data/androidcontrol"
    report = {}
    for setting in ("low", "high"):
        output = output_root / f"{setting}_sample.jsonl"
        with output.open("w") as handle:
            for source in rows:
                sample = source[setting]
                image_hash = image_sha256(sample)
                image_path = output_root / "images" / setting / f"{source['id']}.png"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                image = Image.open(BytesIO(sample["image"]["bytes"])).convert("RGB")
                image.save(image_path, format="PNG")
                expected_source = source[f"source_{setting}_sha256"]
                if source_sha256(sample) != expected_source:
                    raise ValueError(f"AC source mismatch: {setting}/{source['id']}")
                artifact = {
                    "stable_index": source["stable_index"],
                    "id": source["id"],
                    "episode_id": source["episode_id"],
                    "setting": setting,
                    "source_index": source[f"{setting}_index"],
                    "source_sha256": expected_source,
                    "image": str(image_path.relative_to(RUN_DIR.parents[2])),
                    "image_sha256": image_hash,
                    "image_size": list(image.size),
                    "instruction": sample["instruction"],
                    "history": sample.get("history", "None"),
                    "gt_action": sample["gt_action"],
                    "gt_bbox": sample["gt_bbox"],
                    "gt_input_text": sample["gt_input_text"].lower() if sample["gt_action"] == "scroll" else sample["gt_input_text"],
                    "group": sample["group"],
                    "ui_type": sample["ui_type"],
                }
                handle.write(json.dumps(artifact, ensure_ascii=True) + "\n")
        report[setting] = {
            "rows": len(rows),
            "manifest": str(output.relative_to(RUN_DIR.parents[2])),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        }
    result = {"schema_version": 1, "status": "PASS", "settings": report}
    (RUN_DIR / "data/androidcontrol/canonical_manifest.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()