import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--r1",type=Path,required=True); parser.add_argument("--r2",type=Path,required=True); parser.add_argument("--output",type=Path,required=True); args=parser.parse_args(); r1=json.loads(args.r1.read_text()); r2=json.loads(args.r2.read_text())
    if r1["gate"]["R1_pass"]: raise ValueError("R3 requires a separate runtime amendment when R1 passes")
    if r2["status"]!="CANCELLED_R_K1": raise ValueError("R3 cancellation requires R2 R-K1 cancellation")
    result={"schema_version":1,"status":"CANCELLED_R_K1","executed":False,"new_inference_forwards":0,"reason":"R1 showed pass@N growth without B3 realization on the target high-disagreement stratum.","mandatory_controls_not_applicable":{"C_rand":"NOT_RUN_BECAUSE_R3_CANCELLED"},"sources":{"R1_sha256":sha256_file(args.r1),"R2_sha256":sha256_file(args.r2)}}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2,sort_keys=True)+"\n"); print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__": main()
