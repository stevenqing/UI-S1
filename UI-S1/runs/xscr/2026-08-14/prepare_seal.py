import hashlib
import json
import math
import os
from collections import defaultdict
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import yaml
from PIL import Image


RUN_DIR = Path(__file__).resolve().parent
ROOT = RUN_DIR.parents[2]
CONFIG_PATH = RUN_DIR / "configs/xscr_prereg.yaml"
SPEC_PATH = RUN_DIR / "SPEC.md"
AMENDMENT_PATH = RUN_DIR / "AMENDMENT_001_SEAL_ROUNDING.md"
SEAL_PATH = RUN_DIR / "SCREEN_SEAL.json"
MANIFEST_PATH = RUN_DIR / "INPUT_MANIFEST.json"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)


def selected_count(screen_count, fraction):
    count = int((Decimal(str(fraction)) * Decimal(screen_count)).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
    if screen_count > 1:
        count = min(count, screen_count - 1)
    return count


def assignment_score(seed, benchmark, stratum, screen):
    value = f"{seed}|{benchmark}|{stratum}|{screen}".encode()
    return hashlib.sha256(value).hexdigest()


def assign_screens(rows, benchmark, seed, fraction):
    strata = defaultdict(set)
    seen_strata = {}
    for row in rows:
        screen = row["image_sha256"]
        stratum = str(row["fold"])
        previous = seen_strata.setdefault(screen, stratum)
        if previous != stratum:
            raise ValueError(f"screen crosses strata in {benchmark}: {screen}")
        strata[stratum].add(screen)
    assignments = []
    counts = {}
    for stratum, screens in sorted(strata.items()):
        ranked = sorted(screens, key=lambda screen: (assignment_score(seed, benchmark, stratum, screen), screen))
        holdout_count = selected_count(len(ranked), fraction)
        holdout = set(ranked[:holdout_count])
        counts[stratum] = {
            "screens": len(ranked),
            "holdout_screens": holdout_count,
            "exploratory_screens": len(ranked) - holdout_count,
        }
        assignments.extend(
            {
                "image_sha256": screen,
                "side": "holdout" if screen in holdout else "exploratory",
                "stratum": stratum,
            }
            for screen in sorted(ranked)
        )
    return assignments, counts


def resolve_image_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def verify_public_images(rows):
    images = {}
    for row in rows:
        screen = row["image_sha256"]
        path = resolve_image_path(row["image_path"])
        if screen in images:
            if images[screen]["path"] != str(path):
                raise ValueError(f"one screen maps to multiple image paths: {screen}")
            continue
        if not path.is_file() or sha256_file(path) != screen:
            raise ValueError(f"public image mismatch: {screen}")
        with Image.open(path) as image:
            width, height = image.size
        images[screen] = {
            "path": str(path.relative_to(ROOT)),
            "bytes": path.stat().st_size,
            "width": width,
            "height": height,
        }
    return images


def main():
    if SEAL_PATH.exists() or MANIFEST_PATH.exists():
        raise FileExistsError("XSCR screen seal already exists")
    config = yaml.safe_load(CONFIG_PATH.read_text())
    if config["status"] != "POST_SELECTION_FEASIBILITY_PREREGISTERED_BEFORE_SCREEN_SEAL_AND_STATISTICS":
        raise PermissionError("XSCR preregistration status mismatch")
    seed = int(config["prospective_internal_holdout"]["seed"])
    fraction = float(config["prospective_internal_holdout"]["fraction"])

    mind_path = ROOT / config["lanes"]["mind2web"]["public_records"]
    android_path = ROOT / config["lanes"]["androidcontrol_low"]["public_records"]
    mind_all = read_jsonl(mind_path)
    mind = [row for row in mind_all if row.get("sample_key", "").startswith("mind2web/C_uni/")]
    android = read_jsonl(android_path)
    low = [row for row in android if row.get("setting") == "low"]
    high = [row for row in android if row.get("setting") == "high"]
    if len(mind) != 2080 or len(low) != 2000 or len(high) != 2000:
        raise ValueError("XSCR public row-count anchor mismatch")
    if any(len(row["candidates"]) != 12 for row in mind) or any(len(row["candidates"]) != 3 for row in android):
        raise ValueError("XSCR public candidate-count anchor mismatch")

    mind_assignments, mind_counts = assign_screens(mind, "mind2web", seed, fraction)
    android_assignments, android_counts = assign_screens(android, "androidcontrol", seed, fraction)
    low_map = {row["image_sha256"]: row["fold"] for row in low}
    high_map = {row["image_sha256"]: row["fold"] for row in high}
    if low_map != high_map:
        raise ValueError("AndroidControl Low/High screen pairing mismatch")

    mind_images = verify_public_images(mind)
    android_images = verify_public_images(android)
    seal = {
        "schema_version": 1,
        "status": "SEALED_BEFORE_XSCR_Q1_AND_ANY_PRIVATE_LABEL_ACCESS",
        "seed": seed,
        "fraction": fraction,
        "rounding": "decimal_half_up",
        "evidence_status": "POST_SELECTION_FEASIBILITY",
        "confirmatory": False,
        "private_labels_opened": False,
        "q1_computed": False,
        "q2_computed": False,
        "q3_q4_computed": False,
        "benchmarks": {
            "mind2web": {"counts_by_stratum": mind_counts, "assignments": mind_assignments},
            "androidcontrol": {"counts_by_stratum": android_counts, "assignments": android_assignments},
        },
    }
    atomic_json(SEAL_PATH, seal)
    manifest = {
        "schema_version": 1,
        "status": "LOCKED_XSCR_PUBLIC_INPUTS_AND_SCREEN_SEAL",
        "private_labels_opened": False,
        "statistics_computed": False,
        "dependencies": {
            "spec": {"path": str(SPEC_PATH.relative_to(ROOT)), "sha256": sha256_file(SPEC_PATH)},
            "config": {"path": str(CONFIG_PATH.relative_to(ROOT)), "sha256": sha256_file(CONFIG_PATH)},
            "amendment": {"path": str(AMENDMENT_PATH.relative_to(ROOT)), "sha256": sha256_file(AMENDMENT_PATH)},
            "mind2web_public": {"path": str(mind_path.relative_to(ROOT)), "sha256": sha256_file(mind_path), "selected_rows": len(mind)},
            "androidcontrol_public": {"path": str(android_path.relative_to(ROOT)), "sha256": sha256_file(android_path), "selected_rows": len(android)},
            "screen_seal": {"path": str(SEAL_PATH.relative_to(ROOT)), "sha256": sha256_file(SEAL_PATH)},
        },
        "dataset_snapshot": {
            "mind2web": {"images": mind_images, "image_count": len(mind_images), "image_bytes": sum(value["bytes"] for value in mind_images.values())},
            "androidcontrol": {"images": android_images, "image_count": len(android_images), "image_bytes": sum(value["bytes"] for value in android_images.values())},
        },
    }
    atomic_json(MANIFEST_PATH, manifest)
    print(json.dumps({
        "status": manifest["status"],
        "screen_seal_sha256": manifest["dependencies"]["screen_seal"]["sha256"],
        "private_labels_opened": False,
        "statistics_computed": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()