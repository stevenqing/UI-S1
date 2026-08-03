import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--r1",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); r1=json.loads(args.r1.read_text())
    if r1["status"]!="PASS": raise ValueError("R2 requires completed R1")
    if r1["gate"]["R1_pass"]: raise ValueError("R2 is eligible; cancellation artifact is prohibited")
    result={"schema_version":1,"status":"CANCELLED_R_K1","executed":False,"reason":"R1 highest-disagreement B3 N24-minus-N4 did not exceed MDE with positive 99% CI lower bound.","R1_gate":r1["gate"],"mandatory_controls_not_applicable":{"S4_random_budget_multiset":"NOT_RUN_BECAUSE_R2_CANCELLED"},"sources":{"R1_sha256":sha256_file(args.r1)}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
