"""Merge FSDP sharded checkpoint into HuggingFace format.

FSDP saves DTensors sharded on dim=0 across N ranks.
We load all N shards and torch.cat the _local_tensors to reconstruct full weights.
"""
import argparse
import os
import torch
from collections import OrderedDict

def merge_fsdp_checkpoint(ckpt_dir, output_dir, base_model_path):
    actor_dir = os.path.join(ckpt_dir, 'actor')

    # Determine world size from filenames
    shard_files = sorted([
        f for f in os.listdir(actor_dir)
        if f.startswith('model_world_size_') and f.endswith('.pt')
    ], key=lambda x: int(x.split('rank_')[1].split('.')[0]))

    if not shard_files:
        print(f"No shard files found in {actor_dir}")
        return False

    world_size = len(shard_files)
    print(f"Found {world_size} shards in {actor_dir}")

    # Load all shards — extract _local_tensor from DTensors
    all_shards = []
    for sf in shard_files:
        path = os.path.join(actor_dir, sf)
        rank = int(sf.split('rank_')[1].split('.')[0])
        print(f"  Loading rank {rank}: {sf}...")
        shard = torch.load(path, map_location='cpu', weights_only=False)
        # Convert DTensors to regular tensors
        clean = {}
        for k, v in shard.items():
            if hasattr(v, '_local_tensor'):
                clean[k] = v._local_tensor.detach()
            elif isinstance(v, torch.Tensor):
                clean[k] = v.detach()
            else:
                clean[k] = v
        all_shards.append((rank, clean))
        del shard

    # Sort by rank
    all_shards.sort(key=lambda x: x[0])

    # Merge: cat along dim=0 for all keys
    keys = list(all_shards[0][1].keys())
    print(f"Merging {len(keys)} keys across {world_size} ranks...")

    merged_state = OrderedDict()
    for k in keys:
        tensors = [s[1][k] for s in all_shards]
        if isinstance(tensors[0], torch.Tensor) and tensors[0].dim() > 0:
            merged_state[k] = torch.cat(tensors, dim=0).to(torch.bfloat16)
        else:
            # Scalar or non-tensor — just take rank 0
            merged_state[k] = tensors[0]

    del all_shards
    print(f"Merged state dict: {len(merged_state)} keys")

    # Remap FSDP key names to HuggingFace format:
    #   model.language_model.X -> model.X  (language model IS "model" in HF)
    #   model.visual.X -> visual.X
    #   lm_head.X -> lm_head.X (unchanged)
    print("Remapping key names to HuggingFace format...")
    hf_state = OrderedDict()
    for k, v in merged_state.items():
        new_k = k
        if k.startswith('model.language_model.'):
            new_k = 'model.' + k[len('model.language_model.'):]
        elif k.startswith('model.visual.'):
            new_k = 'visual.' + k[len('model.visual.'):]
        hf_state[new_k] = v
    del merged_state

    # Verify a known shape
    test_key = 'model.layers.0.self_attn.q_proj.weight'
    if test_key in hf_state:
        print(f"  Verify: {test_key} shape = {hf_state[test_key].shape} (expected [3584, 3584])")

    # Save
    os.makedirs(output_dir, exist_ok=True)

    from safetensors.torch import save_file
    for k in hf_state:
        if isinstance(hf_state[k], torch.Tensor):
            hf_state[k] = hf_state[k].contiguous()

    print(f"Saving to {output_dir}...")
    save_file(hf_state, os.path.join(output_dir, "model.safetensors"))
    del hf_state

    # Copy config and tokenizer from base model
    from transformers import AutoConfig, AutoProcessor
    config = AutoConfig.from_pretrained(base_model_path, trust_remote_code=True)
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    config.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    print(f"Done! Saved to {output_dir}")
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--base_model', default=None)
    args = parser.parse_args()
    if args.base_model is None:
        args.base_model = os.path.join(args.ckpt_dir, 'actor')
    merge_fsdp_checkpoint(args.ckpt_dir, args.output_dir, args.base_model)
