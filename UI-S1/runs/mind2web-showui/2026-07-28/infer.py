import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch


MODEL_REVISION = "cabec4fcc48d15ffd3efe0b33ea9bc7d41509d60"
PROCESSOR_REVISION = "895c3a49bc3fa70a340399125c650a463535e71c"
CHAT_TEMPLATE = "{% set image_count = namespace(value=0) %}{% set video_count = namespace(value=0) %}{% for message in messages %}<|im_start|>{{ message['role'] }}\n{% if message['content'] is string %}{{ message['content'] }}<|im_end|>\n{% else %}{% for content in message['content'] %}{% if content['type'] == 'image' or 'image' in content or 'image_url' in content %}{% set image_count.value = image_count.value + 1 %}{% if add_vision_id %}Picture {{ image_count.value }}: {% endif %}<|vision_start|><|image_pad|><|vision_end|>{% elif content['type'] == 'video' or 'video' in content %}{% set video_count.value = video_count.value + 1 %}{% if add_vision_id %}Video {{ video_count.value }}: {% endif %}<|vision_start|><|video_pad|><|vision_end|>{% elif 'text' in content %}{{ content['text'] }}{% endif %}{% endfor %}<|im_end|>\n{% endif %}{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"


def load_completed(path: Path) -> set[int]:
    completed = set()
    if not path.exists():
        return completed
    with path.open() as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            index = row["index"]
            if index in completed:
                raise ValueError(f"duplicate index {index} at line {line_number}")
            completed.add(index)
    return completed


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--processor-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--repo-dir", type=Path, default=Path("repos/ShowUI"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-history", type=int, default=2)
    parser.add_argument("--min-visual-tokens", type=int, default=256)
    parser.add_argument("--max-visual-tokens", type=int, default=1344)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU, found {torch.cuda.device_count()}"
        )
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")

    repo_dir = args.repo_dir.resolve()
    sys.path.insert(0, str(repo_dir))
    from data.dataset import collate_fn
    from data.dset_mind2web import Mind2WebDataset
    from model.showui.modeling_showui import ShowUIForConditionalGeneration
    from model.showui.processing_showui import ShowUIProcessor

    processor = ShowUIProcessor.from_pretrained(
        args.processor_dir.resolve(),
        min_pixels=args.min_visual_tokens * 28 * 28,
        max_pixels=args.max_visual_tokens * 28 * 28,
        model_max_length=8192,
        uigraph_train=True,
        uigraph_test=False,
        uigraph_diff=1,
        uigraph_rand=False,
        uimask_pre=True,
        uimask_ratio=0.5,
        uimask_rand=False,
    )
    processor.chat_template = CHAT_TEMPLATE
    processor.tokenizer.chat_template = CHAT_TEMPLATE

    model = ShowUIForConditionalGeneration.from_pretrained(
        args.model_dir.resolve(),
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        attn_implementation="eager",
        device_map="cuda:0",
        lm_skip_layer=[0] * 28,
        lm_skip_ratio=0,
    )
    model.config.use_cache = False
    model.eval()

    dataset = Mind2WebDataset(
        dataset_dir=str(args.data_root.resolve()),
        dataset="mind2web",
        json_data="hf_test_task",
        processor=processor,
        inference=True,
        args_dict={
            "num_turn": 1,
            "num_history": args.num_history,
            "interleaved_history": "tttt",
            "random_sample": False,
        },
    )
    indices = [
        index
        for index in range(len(dataset))
        if index % args.num_shards == args.shard_index
    ]
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
            sample = dataset[index]
            input_dict = collate_fn([sample], processor=processor)
            meta = input_dict.pop("meta_data")[0]
            input_ids_hash = tensor_sha256(input_dict["input_ids"])
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    input_dict[key] = value.to("cuda:0")
            input_dict["pixel_values"] = input_dict["pixel_values"].bfloat16()

            forward_dict = {
                "pixel_values": input_dict["pixel_values"],
                "input_ids": input_dict["input_ids"],
                "labels": input_dict["labels"],
                "image_grid_thw": input_dict["image_sizes"],
                "patch_assign": input_dict["patch_assign"],
                "patch_assign_len": input_dict["patch_assign_len"],
            }
            for key in ("patch_pos", "select_mask"):
                if key in input_dict:
                    forward_dict[key] = input_dict[key]
            with torch.inference_mode():
                generated = model.generate(
                    **forward_dict,
                    max_new_tokens=128,
                    eos_token_id=processor.tokenizer.eos_token_id,
                )
            generated = generated[:, input_dict["input_ids"].shape[1] :]
            response = processor.batch_decode(
                generated,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
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
                "model_revision": MODEL_REVISION,
                "processor_revision": PROCESSOR_REVISION,
                "num_history": args.num_history,
                "min_visual_tokens": args.min_visual_tokens,
                "max_visual_tokens": args.max_visual_tokens,
                "shard_index": args.shard_index,
                "num_shards": args.num_shards,
            }
            output.write(json.dumps(row, ensure_ascii=True) + "\n")
            output.flush()
            os.fsync(output.fileno())


if __name__ == "__main__":
    main()
