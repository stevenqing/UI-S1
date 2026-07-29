import argparse
import json
import logging
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


DATA_REVISION = "17ece8eb89862368edc0cc806acee6fca5163474"
MODEL_REVISION = "848f8100c508e5a742ec2d3ec175b7baa704334c"
TOKENIZER_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--score-file", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()

    random.seed(123)
    np.random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    sys.path.insert(0, str((args.source_dir / "action_prediction").resolve()))
    sys.path.insert(0, str(args.source_dir.resolve()))
    from dataloader import MultiChoiceDataset, get_data_split
    from metric import ActionEvaluatorMultiChoice

    with args.score_file.open("rb") as handle:
        candidate_results = pickle.load(handle)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir.resolve())
    test_data = get_data_split(
        str(args.data_dir.resolve()),
        "test_task/*.json",
        candidate_results=candidate_results,
    )
    if len(test_data) != 2094:
        raise ValueError(f"expected 2094 test_task actions, found {len(test_data)}")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must be in [0, num-shards)")
    if args.limit is not None:
        test_data = test_data.select(range(min(args.limit, len(test_data))))

    shuffle_orders = []
    global_rng = random.Random(123)
    selected_indices = []
    for index, sample in enumerate(test_data):
        positive_ids = [
            candidate["backend_node_id"]
            for candidate in sample["pos_candidates"]
            if candidate["rank"] < 50
        ]
        if positive_ids:
            negative_ids = [
                candidate["backend_node_id"]
                for candidate in sample["neg_candidates"]
                if candidate["rank"] < 50
            ]
            order = positive_ids + negative_ids
            global_rng.shuffle(order)
        else:
            order = None
        if index % args.num_shards == args.shard_index:
            selected_indices.append(index)
            if order is not None:
                shuffle_orders.append(order)
    test_data = test_data.select(selected_indices)

    original_shuffle = random.shuffle
    order_iterator = iter(shuffle_orders)

    def replay_global_shuffle(values: list) -> None:
        expected = next(order_iterator)
        if sorted(values) != sorted(expected):
            raise ValueError("candidate set does not match precomputed global shuffle")
        values[:] = expected

    random.shuffle = replay_global_shuffle
    dataset = MultiChoiceDataset(
        test_data,
        tokenizer,
        neg_ratio=0.2,
        num_candidates=5,
        max_context_len=512,
        mode="multichoice",
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_dir.resolve(),
        torch_dtype=torch.bfloat16,
    ).to("cuda").eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    evaluator = ActionEvaluatorMultiChoice(tokenizer)
    with torch.inference_mode():
        try:
            result = evaluator.evaluate_dataset(
                dataset,
                model,
                batch_size=1,
                top_k=50,
                output_path=str(args.output_dir.resolve()),
                name="test_task",
            )
        finally:
            random.shuffle = original_shuffle
    provenance = {
        "status": "COMPLETE" if args.num_shards == 1 and len(test_data) == 2094 else "PARTIAL",
        "actions": len(test_data),
        "episodes": len(set(test_data["annotation_id"])),
        "model": "osunlp/MindAct_ActionPrediction_flan-t5-xl",
        "model_revision": MODEL_REVISION,
        "tokenizer": "google/flan-t5-xl",
        "tokenizer_revision": TOKENIZER_REVISION,
        "dataset": "osunlp/Mind2Web",
        "dataset_revision": DATA_REVISION,
        "split": "test_task",
        "candidate_scores": "scores_all_data.pkl",
        "top_k": 50,
        "seed": 123,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "global_indices": selected_indices,
        "global_shuffle_replayed": True,
        "result": result,
    }
    (args.output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps(provenance, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()