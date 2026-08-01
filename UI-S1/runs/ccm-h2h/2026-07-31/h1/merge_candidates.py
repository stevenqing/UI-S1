import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


SEEDS = (20260731, 20260732, 20260733)


def candidate_hash(candidates):
    payload = json.dumps(candidates, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def seeded_candidates(row, seed):
    candidates = row["candidates"]
    full = candidates[0]
    subimages = candidates[1:]
    if len(subimages) < 9 or len(subimages) > 18:
        raise ValueError(f"MDE superset outside [9,18]: {row['id']}")
    row_index = row["stable_index"]
    rng = np.random.default_rng(np.random.SeedSequence([seed, row_index]))
    coverage = np.asarray([candidate["coverage"] for candidate in subimages], dtype=np.float64)
    gumbel = rng.gumbel(size=len(subimages))
    score = np.log(coverage + 1.0) + 0.25 * gumbel
    order = sorted(range(len(subimages)), key=lambda index: (-score[index], index))
    return [full] + [subimages[index] for index in order[:9]]


def write_parquet(rows, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output, compression="zstd")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=8)
    args = parser.parse_args()
    by_id = {}
    for shard in range(args.num_shards):
        path = args.shard_root / f"shard-{shard}.jsonl"
        if not path.exists():
            raise FileNotFoundError(path)
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in by_id:
                raise ValueError(f"duplicate H1 candidate id: {row['id']}")
            if row["shard_index"] != shard or row["num_shards"] != args.num_shards:
                raise ValueError("H1 shard metadata mismatch")
            if row["candidate_count"] < 10 or row["candidate_count"] > 19:
                raise ValueError(f"H1 requires 10-19 superset candidates: {row['id']}")
            if len(row["candidates"]) != row["candidate_count"]:
                raise ValueError(f"H1 candidate count metadata mismatch: {row['id']}")
            stages = [candidate["stage"] for candidate in row["candidates"]]
            expected_stages = ["full_image"] + [f"subimage_{index}" for index in range(1, len(row["candidates"]))]
            if stages != expected_stages:
                raise ValueError(f"H1 candidate stage order mismatch: {row['id']}")
            regions = [tuple(candidate["region"]) for candidate in row["candidates"][1:]]
            if len(regions) != len(set(regions)):
                raise ValueError(f"H1 duplicate subimage regions: {row['id']}")
            if candidate_hash(row["candidates"]) != row["candidate_sha256"]:
                raise ValueError(f"H1 candidate source hash mismatch: {row['id']}")
            by_id[row["id"]] = row
    if len(by_id) != 1581:
        raise ValueError(f"H1 merge requires 1,581 rows, found {len(by_id)}")
    ordered = [by_id[key] for key in sorted(by_id)]
    for stable_index, row in enumerate(ordered):
        row["stable_index"] = stable_index

    manifest = {
        "status": "PASS",
        "rows": len(ordered),
        "source_candidate_count_distribution": dict(sorted(Counter(row["candidate_count"] for row in ordered).items())),
        "outputs": {},
    }
    for count in (2, 4, 10):
        derived = []
        for row in ordered:
            candidates = row["candidates"][:count]
            derived.append({
                **{key: value for key, value in row.items() if key not in {"candidates", "candidate_sha256", "candidate_count"}},
                "candidates": candidates,
                "candidate_count": count,
                "candidate_sha256": candidate_hash(candidates),
                "derivation": f"official_ordered_prefix_N{count}",
            })
        path = args.output_dir / f"candidates_N{count}.parquet"
        write_parquet(derived, path)
        manifest["outputs"][f"N{count}"] = {
            "path": str(path),
            "rows": len(derived),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    for seed in SEEDS:
        derived = []
        for row in ordered:
            candidates = seeded_candidates(row, seed)
            derived.append({
                **{key: value for key, value in row.items() if key not in {"candidates", "candidate_sha256", "candidate_count"}},
                "candidates": candidates,
                "candidate_count": 10,
                "candidate_sha256": candidate_hash(candidates),
                "derivation": f"gumbel_proposal_seed_{seed}",
            })
        path = args.output_dir / f"candidates_N10_seed{seed}.parquet"
        write_parquet(derived, path)
        manifest["outputs"][f"N10_seed{seed}"] = {
            "path": str(path),
            "rows": len(derived),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    manifest_path = args.output_dir / "candidate_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
