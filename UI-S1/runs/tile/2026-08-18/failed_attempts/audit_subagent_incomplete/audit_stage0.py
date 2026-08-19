import json
import hashlib
from pathlib import Path
import numpy as np

# Use the virtual environment Python or standard Python with packages
ROOT = Path("/home/aiscuser/UI-S1/UI-S1")
STAGE0_DIR = ROOT / "runs/tile/2026-08-18"
STAGE0_JSON_PATH = STAGE0_DIR / "STAGE0.json"
PAIRS_PATH = STAGE0_DIR / "raw/eccentricity_pairs.jsonl"
CURVES_PATH = STAGE0_DIR / "raw/fold_curves.jsonl"
SCORES_PATH = STAGE0_DIR / "raw/row_scores.jsonl"

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]

# 1. Load STAGE0.json
stage0 = json.loads(STAGE0_JSON_PATH.read_text())

# 2. Check metadata
print("--- Check (1) ---")
for file_key, path in [("pairs", PAIRS_PATH), ("curves", CURVES_PATH), ("row_scores", SCORES_PATH)]:
    meta = stage0[file_key]
    n_rows = len(read_jsonl(path))
    file_bytes = path.stat().st_size
    file_sha = sha256_file(path)
    print(f"{file_key}:")
    print(f"  Expected (JSON): rows={meta['rows']}, bytes={meta['bytes']}, sha256={meta['sha256']}")
    print(f"  Actual (File):   rows={n_rows}, bytes={file_bytes}, sha256={file_sha}")
    match = (n_rows == meta['rows']) and (file_bytes == meta['bytes']) and (file_sha == meta['sha256'])
    print(f"  Match: {match}")


# 3. Check expected counts and uniqueness
print("\n--- Check (2) ---")
pairs_rows = read_jsonl(PAIRS_PATH)
curves_rows = read_jsonl(CURVES_PATH)
scores_rows = read_jsonl(SCORES_PATH)

print(f"Pairs count: {len(pairs_rows)} (Expected 12577)")
print(f"Curves count: {len(curves_rows)} (Expected 200)")
print(f"Scores count: {len(scores_rows)} (Expected 15810)")

# Check uniqueness of (row_id, N) per phase
for phase in ["inner_validation", "outer_test"]:
    phase_rows = [r for r in scores_rows if r["phase"] == phase]
    keys = [(r["row_id"], r["N"]) for r in phase_rows]
    unique_keys = set(keys)
    print(f"Phase {phase}: {len(phase_rows)} rows, {len(unique_keys)} unique (row_id, N)")
    # Group by N
    from collections import Counter
    c = Counter([r["N"] for r in phase_rows])
    print(f"  Count by N: {dict(c)}")


# 4. Check (3) outer_fold matches each row's known fold and curve records fit only non-test folds, using available fields/artifacts to detect leakage
print("\n--- Check (3) ---")
# Let's inspect fold values in scores_rows and compare with what's in eccentricity_pairs or cover
# Let's load COVER
COVER_PATH = ROOT / "runs/cover/2026-08-16/raw/arm_a_rows.jsonl"
cover_data = {r["row_id"]: r for r in read_jsonl(COVER_PATH)}

mismatch_fold = 0
for r in scores_rows:
    row_fold = cover_data[r["row_id"]]["fold"]
    if r["outer_fold"] != row_fold:
        # In inner_validation, does outer_fold match? Let's check stage0.py logic:
        # inner_validation is for cover[row_id]["fold"] == inner_fold where inner_fold = (outer_fold + 1) % 5
        # outer_test is for cover[row_id]["fold"] == outer_fold
        if r["phase"] == "outer_test" and r["outer_fold"] != row_fold:
            mismatch_fold += 1
        elif r["phase"] == "inner_validation" and (r["outer_fold"] + 1) % 5 != row_fold:
            mismatch_fold += 1

print(f"Fold status mismatches based on stage0.py logic: {mismatch_fold}")

# Check curves "fit_folds" mapping:
# For outer_fold in range(5), "inner_train" curves fit on inner_train_folds (which does not include outer_fold or inner_fold = list(set(range(5)) - {outer_fold, inner_fold}))
# "outer_development" curves fit on development_folds (which is set(range(5)) - {outer_fold})
leakage_curves = 0
for curve in curves_rows:
    outer_fold = curve["outer_fold"]
    phase = curve["phase"]
    fit_folds = curve["fit_folds"]
    if phase == "inner_train":
        # inner_fold is (outer_fold + 1) % 5
        inner_fold = (outer_fold + 1) % 5
        # outer_fold and inner_fold must not be in fit_folds
        if outer_fold in fit_folds or inner_fold in fit_folds:
            leakage_curves += 1
    elif phase == "outer_development":
        if outer_fold in fit_folds:
            leakage_curves += 1

print(f"Leakage in curves (test/val folds included in fitting): {leakage_curves}")

# 5. Check (4) & (5) & (6) & (7)
# Let's write the computation code for:
# - fixed-N V-only repair/damage/net
# - C-uni contextual repair/damage/net
# - selected N per fold and selected ledgers from raw scores
# - T-G1, T-G2 ratio, T-K5
# - bootstrap point estimates and reps
# - domain counts: V-only original correct, C-uni original correct, crop-covered

# Let's load values
outer_rows = [r for r in scores_rows if r["phase"] == "outer_test"]

print("\n--- Check (4) & (7) ---")
# Check domain counts
# V-only original correct: unique row_ids where phase == outer_test, N == 4 and V_only_B3_correct is True. (Since each N has same rows, let's look at N=11 or N=4)
n11_outer = [r for r in outer_rows if r["N"] == 11]
v_only_orig_correct_cnt = sum(1 for r in n11_outer if r["V_only_B3_correct"])
c_uni_orig_correct_cnt = sum(1 for r in n11_outer if r["C_uni_B3_correct"])
crop_covered_cnt = sum(1 for r in n11_outer if r["crop_covered"])
print(f"V-only original correct rows (expected 950): {v_only_orig_correct_cnt}")
print(f"C-uni original correct rows (expected 1007): {c_uni_orig_correct_cnt}")
print(f"crop-covered rows (expected 1356): {crop_covered_cnt}")

# Correct quantification for C-uni original correct contextual summary for N=11
# Let's count actual recomputed C_uni_expected_repair, contextual summary for rows where C_uni_B3_correct is True
c_uni_subgroup_n11 = [r for r in n11_outer if r["C_uni_B3_correct"]]
print("N=11 C_uni subgroup count:", len(c_uni_subgroup_n11))
recomputed_c_uni_subgroup_repair = sum(r["C_uni_expected_repair"] for r in c_uni_subgroup_n11)
recomputed_c_uni_subgroup_damage = sum(r["C_uni_expected_damage"] for r in c_uni_subgroup_n11)
recomputed_c_uni_subgroup_net = recomputed_c_uni_subgroup_repair - recomputed_c_uni_subgroup_damage
print(f"N=11 Corrected C-uni original correct contextual subgroup:")
print(f"  Repair: {recomputed_c_uni_subgroup_repair:.6f}")
print(f"  Damage: {recomputed_c_uni_subgroup_damage:.6f}")
print(f"  Net:    {recomputed_c_uni_subgroup_net:.6f}")

# Let's see what is in STAGE0.json for C_uni_original_correct_contextual under fixed_N -> 11
print("STAGE0.json has:")
print(json.dumps(stage0["fixed_N"]["11"]["C_uni_original_correct_contextual"], indent=2))
# End of incomplete audit attempt.
