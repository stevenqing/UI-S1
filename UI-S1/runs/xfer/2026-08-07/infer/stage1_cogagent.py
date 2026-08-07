import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
import yaml
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer


RUN_DIR = Path(__file__).resolve().parents[1]
ROOT = RUN_DIR.parents[2]
sys.path.insert(0, str(RUN_DIR))

from xfer_common import MIND2WEB_ACTIONS, parse_cogagent_response, parse_product_response


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def completed_ids(path):
    if not path.exists(): return set()
    ids=[json.loads(line)["id"] for line in path.read_text().splitlines() if line.strip()]
    if len(ids)!=len(set(ids)): raise ValueError("duplicate CogAgent ids")
    return set(ids)


def prompt_text(roster,row):
    c=roster["mind2web"]["prompt_contract"]
    history="\n".join(str(x.get("step_repr") or x.get("operation") or x) for x in row["step_history"][-c["history_steps"]:]) or "None"
    return c["user_template"].format(task=row["task"],history=history)+"\n(with grounding)"


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,required=True); parser.add_argument("--num-shards",type=int,default=4); parser.add_argument("--shard-index",type=int,required=True); parser.add_argument("--limit",type=int); parser.add_argument("--resume",action="store_true"); args=parser.parse_args()
    roster=yaml.safe_load((RUN_DIR/"configs/xfer_roster.yaml").read_text()); spec=next(x for x in roster["mind2web"]["models"] if x["id"]=="CogAgent-18B")
    rows=[json.loads(x) for x in (RUN_DIR/"data/mind2web/mind2web_test_task.jsonl").read_text().splitlines() if x.strip()]
    indices=list(range(args.shard_index,len(rows),args.num_shards)); indices=indices[:args.limit] if args.limit is not None else indices
    args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.output.exists() and not args.resume: raise FileExistsError(args.output)
    completed=completed_ids(args.output)
    tokenizer=LlamaTokenizer.from_pretrained(ROOT/"runs/xfer/2026-08-07/models/vicuna-7b-v1.5")
    model=AutoModelForCausalLM.from_pretrained(ROOT/spec["local_path"],torch_dtype=torch.bfloat16,low_cpu_mem_usage=True,trust_remote_code=True).to("cuda:0").eval()
    index_hash=sha256_file(ROOT/spec["local_path"]/"model.safetensors.index.json")
    with args.output.open("a",buffering=1) as output:
        for index in indices:
            row=rows[index]
            if row["id"] in completed: continue
            image=Image.open(ROOT/row["image"]).convert("RGB"); query=prompt_text(roster,row)
            conversation=model.build_conversation_input_ids(tokenizer,query=query,history=[],images=[image],template_version="chat")
            inputs={"input_ids":conversation["input_ids"].unsqueeze(0).to("cuda:0"),"token_type_ids":conversation["token_type_ids"].unsqueeze(0).to("cuda:0"),"attention_mask":conversation["attention_mask"].unsqueeze(0).to("cuda:0"),"images":[[conversation["images"][0].to("cuda:0",dtype=torch.bfloat16)]],"cross_images":[[conversation["cross_images"][0].to("cuda:0",dtype=torch.bfloat16)]]}
            with torch.inference_mode(): generated=model.generate(**inputs,max_new_tokens=256,do_sample=False,eos_token_id=tokenizer.eos_token_id,pad_token_id=tokenizer.pad_token_id)
            response=tokenizer.decode(generated[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True)
            try:
                prediction=parse_product_response(response,MIND2WEB_ACTIONS) if response.strip().startswith("{") else parse_cogagent_response(response,MIND2WEB_ACTIONS)
            except (json.JSONDecodeError,TypeError,ValueError): prediction={"action":None,"value":None,"position":None,"parse_ok":False}
            artifact={"stable_index":index,"id":row["id"],"annot_id":row["annot_id"],"action_uid":row["action_uid"],"image_sha256":row["image_sha256"],"model_id":"CogAgent-18B","model_revision":spec["revision"],"model_index_sha256":index_hash,"response":response,"prediction":prediction,"shard_index":args.shard_index,"num_shards":args.num_shards}
            output.write(json.dumps(artifact,ensure_ascii=True)+"\n"); output.flush(); os.fsync(output.fileno())
    print(json.dumps({"status":"PASS","completed":len(completed_ids(args.output))}))


if __name__=="__main__": main()