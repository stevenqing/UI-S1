import hashlib
import json
import zipfile
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
SEED = 20260807
EXPECTED_M2W_METADATA_SHA = "6e86b61ab6b8c657cabadc73de9df1f844dc39e4904228c8b2b5a18b68640d2d"
EXPECTED_M2W_ANNOTATION_SHA = "310200fcfe0c3004acc0d19621e50b2801811291bf8e95f641ed6ddc2e1bb906"
EXPECTED_AC_SHA = {
    "low": "ffb8e19f5091c339aea4060e062cc47405f57da3f35c22699a82390d5769cf47",
    "high": "ec70c99046aa4fb1557c61bf2c5d1266f87d4b9dc879128b2de20bf0aca7c72f",
}


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def extract_mind2web():
    root = RUN_DIR / "data/mind2web"
    annotations_zip = root / "downloads/mind2web_annots.zip"
    images_zip = root / "downloads/mind2web_images.zip"
    extracted = root / "extracted"
    extracted.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(annotations_zip) as archive:
        archive.extractall(extracted / "annotations")
    annotation_candidates = list((extracted / "annotations").rglob("mind2web_data_test_task.json"))
    if len(annotation_candidates) != 1:
        raise ValueError(f"Mind2Web annotation path ambiguity: {annotation_candidates}")
    annotation_path = annotation_candidates[0]
    if sha256_file(annotation_path) != EXPECTED_M2W_ANNOTATION_SHA:
        raise ValueError("Mind2Web annotation hash mismatch")
    metadata_path = root / "hf_test_task_with_thoughts.json"
    if sha256_file(metadata_path) != EXPECTED_M2W_METADATA_SHA:
        raise ValueError("Mind2Web thought metadata hash mismatch")
    metadata = json.loads(metadata_path.read_text())
    annotations = json.loads(annotation_path.read_text())
    expected = {}
    for episode in annotations:
        for index, (step, step_repr) in enumerate(zip(episode["actions"], episode["action_reprs"])):
            if not step.get("bbox"):
                continue
            identity = (episode["annotation_id"], step["action_uid"])
            expected[identity] = {
                "episode": episode,
                "step": step,
                "step_repr": step_repr,
                "step_index": index,
            }
    actual = {(row["annot_id"], row["action_uid"]): row for row in metadata}
    if len(metadata) != 2080 or len(actual) != 2080 or set(actual) != set(expected):
        raise ValueError("Mind2Web identity coverage mismatch")
    required_images = {row["img_url"] for row in metadata}
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(images_zip) as archive:
        members = {Path(name).name: name for name in archive.namelist() if name.lower().endswith(".jpg")}
        missing = sorted(required_images - set(members))
        if missing:
            raise ValueError(f"Mind2Web archive missing images: {missing[:3]}")
        for filename in sorted(required_images):
            destination = image_dir / filename
            if not destination.exists():
                with archive.open(members[filename]) as source, destination.open("wb") as target:
                    target.write(source.read())
    rows = []
    for stable_index, identity in enumerate(sorted(actual)):
        row = actual[identity]
        source = expected[identity]
        episode = source["episode"]
        step = source["step"]
        if row["step"]["bbox"] != step["bbox"] or row["step"]["operation"] != step["operation"]:
            raise ValueError(f"Mind2Web GT mismatch: {identity}")
        if row["task"] != episode["confirmed_task"] or row["step_repr"] != source["step_repr"]:
            raise ValueError(f"Mind2Web task/history mismatch: {identity}")
        image_path = image_dir / row["img_url"]
        rows.append({
            "stable_index": stable_index,
            "id": f"{identity[0]}__{identity[1]}",
            "annot_id": identity[0],
            "action_uid": identity[1],
            "episode_id": identity[0],
            "website": episode.get("website") or episode.get("domain") or identity[0],
            "image": str(image_path.relative_to(ROOT)),
            "image_sha256": sha256_file(image_path),
            "task": row["task"],
            "step_history": row["step_history"],
            "step": row["step"],
            "step_repr": row["step_repr"],
        })
    output = root / "mind2web_test_task.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    return {
        "rows": len(rows),
        "episodes": len({row["episode_id"] for row in rows}),
        "images": len(required_images),
        "output": str(output.relative_to(ROOT)),
        "sha256": sha256_file(output),
    }


def ac_key(row):
    image = row["image"]
    image_bytes = image["bytes"] if isinstance(image, dict) else image
    image_hash = hashlib.sha256(image_bytes).hexdigest()
    bbox = tuple(row.get("gt_bbox") or row.get("bbox") or ())
    action = row.get("gt_action") or row.get("action")
    return image_hash, action, bbox


def pair_androidcontrol(low_rows, high_rows):
    low_groups = defaultdict(list)
    high_groups = defaultdict(list)
    for index, row in enumerate(low_rows):
        low_groups[ac_key(row)].append(index)
    for index, row in enumerate(high_rows):
        high_groups[ac_key(row)].append(index)
    if {key: len(values) for key, values in low_groups.items()} != {key: len(values) for key, values in high_groups.items()}:
        raise ValueError("AndroidControl Low/High identity multisets differ")
    mapping = {}
    conflict_low = set()
    for key, low_indices in low_groups.items():
        by_parameter = defaultdict(deque)
        for high_index in high_groups[key]:
            value = high_rows[high_index].get("gt_input_text", "")
            by_parameter[value].append(high_index)
        unmatched = []
        for low_index in low_indices:
            value = low_rows[low_index].get("gt_input_text", "")
            if by_parameter[value]:
                mapping[low_index] = by_parameter[value].popleft()
            else:
                unmatched.append(low_index)
        remaining = sorted(index for values in by_parameter.values() for index in values)
        if len(unmatched) != len(remaining):
            raise ValueError("AndroidControl duplicate-aware pairing failed")
        for low_index, high_index in zip(sorted(unmatched), remaining):
            mapping[low_index] = high_index
            conflict_low.add(low_index)
    if len(mapping) != 7708 or len(conflict_low) != 58:
        raise ValueError(f"AndroidControl expected 7708 pairs/58 conflicts, found {len(mapping)}/{len(conflict_low)}")
    return mapping, conflict_low


def stratified_sample(clean_indices, rows, size):
    by_action = defaultdict(list)
    for index in clean_indices:
        action = rows[index].get("gt_action") or rows[index].get("action")
        by_action[action].append(index)
    rng = np.random.default_rng(SEED)
    counts = {action: len(values) for action, values in by_action.items()}
    exact = {action: size * count / len(clean_indices) for action, count in counts.items()}
    allocation = {action: int(np.floor(value)) for action, value in exact.items()}
    remainder = size - sum(allocation.values())
    for action in sorted(exact, key=lambda value: (-(exact[value] - allocation[value]), value))[:remainder]:
        allocation[action] += 1
    selected = []
    for action in sorted(by_action):
        values = np.asarray(sorted(by_action[action]), dtype=np.int64)
        selected.extend(map(int, rng.choice(values, size=allocation[action], replace=False)))
    selected.sort()
    if len(selected) != size or len(set(selected)) != size:
        raise ValueError("AndroidControl sample size/uniqueness mismatch")
    return selected, counts, allocation


def prepare_androidcontrol():
    root = RUN_DIR / "data/androidcontrol"
    for setting, digest in EXPECTED_AC_SHA.items():
        if sha256_file(root / f"androidcontrol_{setting}_test.parquet") != digest:
            raise ValueError(f"AndroidControl {setting} data hash mismatch")
    low = pq.read_table(root / "androidcontrol_low_test.parquet").to_pylist()
    high = pq.read_table(root / "androidcontrol_high_test.parquet").to_pylist()
    if len(low) != 7708 or len(high) != 7708:
        raise ValueError("AndroidControl row count mismatch")
    mapping, conflicts = pair_androidcontrol(low, high)
    clean = sorted(set(range(7708)) - conflicts)
    selected, full_counts, sample_counts = stratified_sample(clean, low, 2000)
    records = []
    for stable_index, low_index in enumerate(selected):
        records.append({
            "stable_index": stable_index,
            "id": f"ac_{low_index}",
            "low_index": low_index,
            "high_index": mapping[low_index],
            "gt_action": low[low_index].get("gt_action") or low[low_index].get("action"),
            "source_low_sha256": canonical_hash({key: value for key, value in low[low_index].items() if key != "image"}),
            "source_high_sha256": canonical_hash({key: value for key, value in high[mapping[low_index]].items() if key != "image"}),
        })
    output = root / "subsample.jsonl"
    output.write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in records))
    config = {
        "schema_version": 1,
        "frozen_at": "2026-08-07",
        "status": "FROZEN_BEFORE_INFERENCE",
        "seed": SEED,
        "source_rows_per_setting": 7708,
        "parameter_conflicts_excluded": 58,
        "clean_rows_per_setting": len(clean),
        "sample_rows_per_setting": len(selected),
        "stratification": "proportional_by_low_gt_action_largest_remainder",
        "full_clean_action_counts": dict(sorted(full_counts.items())),
        "sample_action_counts": dict(sorted(sample_counts.items())),
        "row_manifest": str(output.relative_to(ROOT)),
        "row_manifest_sha256": sha256_file(output),
        "low_data_sha256": EXPECTED_AC_SHA["low"],
        "high_data_sha256": EXPECTED_AC_SHA["high"],
    }
    config_path = RUN_DIR / "configs/ac_subsample.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return config


def main():
    result = {
        "schema_version": 1,
        "status": "PASS",
        "mind2web": extract_mind2web(),
        "androidcontrol": prepare_androidcontrol(),
    }
    output = RUN_DIR / "data_manifest.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
