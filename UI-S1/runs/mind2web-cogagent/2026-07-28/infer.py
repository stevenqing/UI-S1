import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForCausalLM, LlamaTokenizer

from common import (
    MODEL_NAME,
    MODEL_REVISION,
    TOKENIZER_NAME,
    TOKENIZER_REVISION,
    expected_answer,
    format_prompt,
    prompt_sha256,
    read_json,
    read_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    metadata = read_json(args.metadata)
    if len(metadata) != 2080:
        raise ValueError(f"expected 2080 prepared rows, found {len(metadata)}")
    indices = [index for index in range(len(metadata)) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        indices = indices[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {output_path}")
    existing = read_jsonl(output_path) if args.resume else []
    completed = {row["index"] for row in existing}
    if len(completed) != len(existing):
        raise ValueError("duplicate indices in existing predictions")

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_dir.resolve())
    model = AutoModelForCausalLM.from_pretrained(
        args.model_dir.resolve(),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0").eval()

    with output_path.open("a", buffering=1) as output:
        for index in indices:
            if index in completed:
                continue
            sample = metadata[index]
            image_path = args.image_root / sample["img_url"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            image = Image.open(image_path).convert("RGB")
            query = format_prompt(sample)
            conversation = model.build_conversation_input_ids(
                tokenizer,
                query=query,
                history=[],
                images=[image],
                template_version="chat",
            )
            inputs = {
                "input_ids": conversation["input_ids"].unsqueeze(0).to("cuda:0"),
                "token_type_ids": conversation["token_type_ids"].unsqueeze(0).to("cuda:0"),
                "attention_mask": conversation["attention_mask"].unsqueeze(0).to("cuda:0"),
                "images": [[conversation["images"][0].to("cuda:0", dtype=torch.bfloat16)]],
                "cross_images": [[conversation["cross_images"][0].to("cuda:0", dtype=torch.bfloat16)]],
            }
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id,
                )
            generated = generated[:, inputs["input_ids"].shape[1] :]
            response = tokenizer.decode(generated[0], skip_special_tokens=True)
            row = {
                "index": index,
                "annot_id": sample["annot_id"],
                "action_uid": sample["action_uid"],
                "split": sample["split"],
                "image": sample["img_url"],
                "answer": expected_answer(sample),
                "bbox": sample["step"]["bbox"],
                "image_size": sample["img_size"],
                "response": response,
                "prompt_sha256": prompt_sha256(sample),
                "model_name": MODEL_NAME,
                "model_revision": MODEL_REVISION,
                "tokenizer_name": TOKENIZER_NAME,
                "tokenizer_revision": TOKENIZER_REVISION,
                "prompt_contract": "official_generic_single_round_with_grounding",
                "coordinate_space": "bbox_0_1000",
                "generation": "greedy",
                "max_new_tokens": args.max_new_tokens,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            output.write(json.dumps(row, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()