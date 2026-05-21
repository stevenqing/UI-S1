"""Entry point for cooperative LoRA GRPO training with verl.

Usage:
  python v10/verl_recipe/main_coop_grpo.py --config v10/verl_recipe/config/coop_grpo_ac.yaml

Or via Hydra:
  python v10/verl_recipe/main_coop_grpo.py \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-VL-7B-Instruct \
    data.train_files=data/ac_train.parquet \
    trainer.n_gpus_per_node=4 trainer.nnodes=2
"""

import os
import sys

import hydra
import ray
from omegaconf import OmegaConf


def run_coop_grpo(config):
    """Initialize Ray, create workers, and start cooperative GRPO training."""
    if not ray.is_initialized():
        ray_kwargs = {
            "runtime_env": {
                "env_vars": {
                    "TOKENIZERS_PARALLELISM": "true",
                    "NCCL_DEBUG": "WARN",
                    "VLLM_LOGGING_LEVEL": "WARN",
                    "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
                }
            },
        }
        # Only set num_cpus when NOT connecting to existing cluster
        if "RAY_ADDRESS" not in os.environ:
            ray_kwargs["num_cpus"] = config.ray_init.get("num_cpus", os.cpu_count())
        ray.init(**ray_kwargs)

    runner = TaskRunner.remote()
    ray.get(runner.run.remote(config))


@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        from pprint import pprint

        from verl.utils.fs import copy_to_local
        from verl.utils import hf_processor, hf_tokenizer

        OmegaConf.resolve(config)
        pprint(OmegaConf.to_container(config, resolve=True))

        # Load tokenizer and processor
        local_path = copy_to_local(
            config.actor_rollout_ref.model.path,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust_remote_code = config.data.get("trust_remote_code", False)
        tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        processor = hf_processor(local_path, trust_remote_code=trust_remote_code, use_fast=True)

        # Worker classes
        from verl.single_controller.ray import RayWorkerGroup
        from v10.verl_recipe.coop_fsdp_workers import CoopActorRolloutRefWorker

        from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(CoopActorRolloutRefWorker),
        }

        # Resource pools
        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [int(config.trainer.n_gpus_per_node)] * int(config.trainer.nnodes),
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
        }

        # Note: ref policy is NOT a separate worker when using LoRA.
        # verl sets ref_in_actor=True when lora_rank>0, which makes
        # compute_ref_log_prob use disable_adapter() on the actor worker.

        # Reward function
        from verl.trainer.ppo.reward import load_reward_manager
        reward_fn = load_reward_manager(
            config, tokenizer, num_examine=0,
            **config.reward_model.get("reward_kwargs", {}),
        )
        val_reward_fn = load_reward_manager(
            config, tokenizer, num_examine=1,
            **config.reward_model.get("reward_kwargs", {}),
        )

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping,
        )

        # Dataset
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler
        from verl.utils.dataset.rl_dataset import collate_fn

        train_dataset = create_rl_dataset(
            config.data.train_files, config.data, tokenizer, processor,
        )
        val_dataset = create_rl_dataset(
            config.data.val_files, config.data, tokenizer, processor,
        )
        train_sampler = create_rl_sampler(config.data, train_dataset)

        # Trainer
        from v10.verl_recipe.coop_ray_trainer import CoopRayTrainer

        trainer = CoopRayTrainer(
            config=config,
            tokenizer=tokenizer,
            processor=processor,
            role_worker_mapping=role_worker_mapping,
            resource_pool_manager=resource_pool_manager,
            ray_worker_group_cls=RayWorkerGroup,
            reward_fn=reward_fn,
            val_reward_fn=val_reward_fn,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            collate_fn=collate_fn,
            train_sampler=train_sampler,
        )

        trainer.init_workers()
        trainer.fit()


@hydra.main(config_path="config", config_name="coop_grpo_ac", version_base=None)
def main(config):
    run_coop_grpo(config)


if __name__ == "__main__":
    main()
