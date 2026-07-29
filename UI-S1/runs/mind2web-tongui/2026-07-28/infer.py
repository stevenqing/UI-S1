import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from torch.utils.data._utils.collate import default_collate


def load_completed(path: Path) -> set[int]:
    completed = set()
    if not path.exists():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            index = json.loads(line)["index"]
            if index in completed:
                raise ValueError(f"duplicate index {index} at line {line_number}")
            completed.add(index)
    return completed


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/evaluation_data"))
    parser.add_argument("--repo-dir", type=Path, default=Path("repos/TongUI-agent"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected exactly one visible GPU, found {torch.cuda.device_count()}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    sys.path.insert(0, str(args.repo_dir.resolve()))
    from tongui.data.dset_mind2web import Mind2WebDataset
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(
        args.model_dir.resolve(),
        min_pixels=256 * 28 * 28,
        max_pixels=1344 * 28 * 28,
        model_max_length=8196,
        use_fast=False,
    )
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model_dir.resolve(),
        device_map="cuda:0",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    dataset = Mind2WebDataset(
        str(args.data_root.resolve()),
        "Mind2Web",
        "hf_test_task_with_thoughts",
        processor,
        inference=True,
        args_dict={"num_history": 2, "interleaved_history": "vtvt", "version": "v2"},
    )
    indices = [index for index in range(len(dataset)) if index % args.num_shards == args.shard_index]
    if args.limit is not None:
        indices = indices[: args.limit]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "predictions.jsonl"
    if output_path.exists() and not args.resume:
        raise FileExistsError(f"output exists without --resume: {output_path}")
    completed = load_completed(output_path) if args.resume else set()

    with output_path.open("a", buffering=1) as output:
        for index in indices:
            if index in completed:
                continue
            data_dict, meta = dataset[index]
            input_dict = default_collate([data_dict])
            input_ids_hash = tensor_sha256(input_dict["input_ids"])
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to("cuda:0")
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()
            forward_dict = {
                "pixel_values": input_dict["pixel_values"],
                "input_ids": input_dict["input_ids"],
                "labels": input_dict["labels"],
                "image_grid_thw": input_dict["image_sizes"].squeeze(dim=0),
            }
            for key in ("patch_assign", "patch_assign_len", "patch_pos", "select_mask"):
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
            with torch.inference_mode():
                generated = model.generate(
                    **forward_dict,
                    max_new_tokens=128,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    do_sample=False,
                )
            generated = generated[:, input_dict["input_ids"].shape[1] :]
            response = processor.batch_decode(
                generated, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            row = {
                "index": index,
                "annot_id": meta["annot_id"],
                "action_uid": meta["action_uid"],
                "split": meta["split"],
                "image": meta["img_url"],
                "answer": meta["answer"],
                "bbox": meta["step"]["bbox"],
                "image_size": meta["img_size"],
                "response": response,
                "input_ids_sha256": input_ids_hash,
                "model_name": args.model_name,
                "model_revision": args.model_revision,
                "version": "v2",
                "num_history": 2,
                "interleaved_history": "vtvt",
                "min_visual_tokens": 256,
                "max_visual_tokens": 1344,
                "attention_backend": "flash_attention_2",
                "generation": "greedy",
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            output.write(json.dumps(row, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()
