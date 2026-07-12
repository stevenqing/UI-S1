#!/usr/bin/env python3
"""Evaluate a GT-action-free visual-transition gate for revision data curation."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")


def key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["correction_id"]), int(row["step_idx"])


def split_name(episode_id: str, seed: int) -> str:
    value = int(hashlib.sha256(f"{seed}:{episode_id}".encode()).hexdigest()[:16], 16) % 10
    return "train" if value < 8 else "dev" if value == 8 else "test"


def coordinate(action: Any) -> tuple[float, float] | None:
    if not isinstance(action, Mapping) or str(action.get("action")) != "click":
        return None
    value = action.get("coordinate")
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        x, y = float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None
    return (x, y) if math.isfinite(x) and math.isfinite(y) else None


@lru_cache(maxsize=16)
def transition_map(before_path: str, after_path: str) -> tuple[np.ndarray, float, float, float]:
    before = np.asarray(Image.open(before_path).convert("RGB"), dtype=np.float32)
    after_image = Image.open(after_path).convert("RGB")
    if after_image.size != (before.shape[1], before.shape[0]):
        after_image = after_image.resize((before.shape[1], before.shape[0]), Image.BILINEAR)
    after = np.asarray(after_image, dtype=np.float32)
    diff = np.mean(np.abs(after - before), axis=2)
    total = float(diff.sum())
    height, width = diff.shape
    if total > 1e-6:
        yy, xx = np.indices(diff.shape)
        cx = float((diff * xx).sum() / total)
        cy = float((diff * yy).sum() / total)
    else:
        cx, cy = width / 2.0, height / 2.0
    return diff, total, cx, cy


def coordinate_features(diff: np.ndarray, total: float, cx: float, cy: float, coord: tuple[float, float]) -> dict[str, float]:
    height, width = diff.shape
    x = min(max(coord[0], 0.0), width - 1.0); y = min(max(coord[1], 0.0), height - 1.0)
    output = {}
    for radius in (32, 64, 96):
        x0=max(0,int(x-radius));x1=min(width,int(x+radius+1));y0=max(0,int(y-radius));y1=min(height,int(y+radius+1))
        local=float(diff[y0:y1,x0:x1].sum())
        output[f"mass_r{radius}"] = local / max(total, 1e-6)
    output["centroid_proximity"] = 1.0 - min(1.0, math.hypot(x-cx,y-cy)/math.hypot(width,height))
    return output


def utility(row: Mapping[str, Any]) -> int:
    return 1 if row["outcome"] == "rescue" else -1 if row["outcome"] == "regress" else 0


def gate_stats(rows: Sequence[Mapping[str, Any]], feature: str, threshold: float) -> dict[str, Any]:
    selected=[row for row in rows if row.get("eligible") and float(row[feature])>=threshold]
    rescue=sum(row["outcome"]=="rescue" for row in selected);regress=sum(row["outcome"]=="regress" for row in selected)
    baseline=sum(row["student_correct"] for row in rows)
    return {"feature":feature,"threshold":threshold,"rows":len(rows),"accepted":len(selected),"coverage":len(selected)/len(rows),"eligible_coverage":len(selected)/max(1,sum(bool(r.get('eligible')) for r in rows)),"rescue":rescue,"regress":regress,"neutral":len(selected)-rescue-regress,"rescue_precision":rescue/max(1,len(selected)),"population_net_utility":(rescue-regress)/len(rows),"fallback_student_accuracy":(baseline+rescue-regress)/len(rows)}


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--causal-input",required=True);parser.add_argument("--student-eval",required=True);parser.add_argument("--trajectories",required=True);parser.add_argument("--output-dir",required=True);parser.add_argument("--min-dev-accepted",type=int,default=10);parser.add_argument("--min-transition-mass",type=float,default=1000.0);parser.add_argument("--seed",type=int,default=42);parser.add_argument("--include-splits",nargs="+",default=["dev","test"])
    args=parser.parse_args()
    source_rows=read_jsonl(Path(args.causal_input));source={key(r):r for r in source_rows};students={key(r):r for r in read_jsonl(Path(args.student_eval))}
    trajectories={str(r['trajectory_id']):r for r in read_jsonl(Path(args.trajectories))}
    rows=[];included_splits=set(args.include_splits)
    for paired,src in source.items():
        row_split=split_name(str(src['episode_id']),args.seed)
        if row_split not in included_splits:
            continue
        stu=students[paired];trajectory=trajectories[str(src['trajectory_id'])];idx=int(src['step_idx']);steps=sorted(trajectory['steps'],key=lambda x:int(x['step_idx']))
        rev_coord=coordinate(src.get('revision_action'));stu_coord=coordinate(stu.get('student_action'));next_image=str(steps[idx+1]['screenshot']) if idx+1<len(steps) else None
        item={"correction_id":src['correction_id'],"trajectory_id":src['trajectory_id'],"episode_id":src['episode_id'],"step_idx":idx,"split":row_split,"student_correct":bool(stu['student_correct']),"revision_correct":bool(src['revision_correct']),"outcome":"rescue" if (not stu['student_correct'] and src['revision_correct']) else "regress" if (stu['student_correct'] and not src['revision_correct']) else "both_correct" if (stu['student_correct'] and src['revision_correct']) else "both_wrong","before_image":src['image'],"after_image":next_image,"eligible":False}
        if next_image and rev_coord and stu_coord:
            diff,total,cx,cy=transition_map(str(src['image']),next_image)
            if total>=args.min_transition_mass:
                rev=coordinate_features(diff,total,cx,cy,rev_coord);student=coordinate_features(diff,total,cx,cy,stu_coord)
                item['eligible']=True;item['transition_mass']=total
                for name in rev:item[f"delta_{name}"]=rev[name]-student[name]
        rows.append(item)
    splits={name:[r for r in rows if r['split']==name] for name in ('train','dev','test')}
    features=("delta_mass_r32","delta_mass_r64","delta_mass_r96","delta_centroid_proximity")
    candidates=[]
    for feature in features:
        for threshold in sorted({float(r[feature]) for r in splits['dev'] if r.get('eligible')},reverse=True):
            result=gate_stats(splits['dev'],feature,threshold)
            if result['accepted']>=args.min_dev_accepted and result['population_net_utility']>0:candidates.append(result)
    candidates.sort(key=lambda r:(r['population_net_utility'],r['rescue_precision'],r['accepted']),reverse=True)
    selected_dev=candidates[0] if candidates else None
    if selected_dev:
        locked=gate_stats(splits['test'],selected_dev['feature'],selected_dev['threshold']);gate='POSITIVE_TEST_UTILITY' if locked['population_net_utility']>0 else 'NO_TEST_UTILITY'
    else:
        locked=gate_stats(splits['test'],features[0],1e30);gate='NO_POSITIVE_DEV_THRESHOLD'
    out=Path(args.output_dir);write_jsonl(out/'features.jsonl',rows)
    summary={"version":"visual-transition-revision-gate-v1","uses_gt_action":False,"uses_future_screenshot":True,"include_splits":args.include_splits,"min_transition_mass":args.min_transition_mass,"split_rows":{k:len(v) for k,v in splits.items()},"eligible_rows":{k:sum(r['eligible'] for r in v) for k,v in splits.items()},"selected_dev_gate":selected_dev,"locked_test":locked,"gate":gate,"top_dev_gates":candidates[:20]}
    write_json(out/'summary.json',summary)
    print(json.dumps({"gate":gate,"selected_dev":selected_dev,"test":locked,"output":str(out)},indent=2))


if __name__=="__main__":main()
