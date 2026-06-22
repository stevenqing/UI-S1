#!/usr/bin/env python3
"""Convert verifier SFT DCP checkpoints to HF/PEFT LoRA adapter checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer, create_sft_dataset
from verl.utils import hf_tokenizer
from verl.utils.device import get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fs import copy_to_local


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert DCP checkpoint to HF/PEFT LoRA adapter format")
    parser.add_argument("--config", required=True, help="Path to verifier_agent_sft.yaml")
    parser.add_argument("--dcp-path", required=True, help="Path to global_step_* DCP checkpoint")
    parser.add_argument("--hf-path", required=True, help="Output HF/PEFT LoRA adapter checkpoint directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parent.parent
    base_config = OmegaConf.load(project_root / "verl" / "trainer" / "config" / "sft_trainer.yaml")
    run_config = OmegaConf.load(args.config)
    config = OmegaConf.merge(base_config, run_config)

    device_name = get_device_name()
    _, _, world_size = initialize_global_process_group()
    device_mesh = init_device_mesh(device_type=device_name, mesh_shape=(world_size,), mesh_dim_names=("fsdp",))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
        mesh_dim_names=("dp", "sp"),
    )

    local_model_path = copy_to_local(src=config.model.partial_pretrain, verbose=True)
    tokenizer = hf_tokenizer(local_model_path, trust_remote_code=config.model.trust_remote_code)
    train_dataset = create_sft_dataset(config.data.train_files, config.data, tokenizer)
    val_dataset = create_sft_dataset(config.data.val_files, config.data, tokenizer)
    trainer = FSDPSFTTrainer(
        config=config,
        device_mesh=device_mesh,
        ulysses_device_mesh=ulysses_device_mesh,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
    )
    trainer.convert_dcp_to_hf(str(Path(args.dcp_path)), str(Path(args.hf_path)))
    destroy_global_process_group()


if __name__ == "__main__":
    main()