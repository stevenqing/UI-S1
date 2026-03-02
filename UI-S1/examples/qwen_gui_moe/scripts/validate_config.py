#!/usr/bin/env python3
"""
配置验证脚本

用于验证 MoE 训练配置是否正确加载，特别是确保保守设置生效。

运行方式:
    python validate_config.py --config examples/qwen_gui_moe/config/traj_grpo_moe.yaml
"""

import sys
import os
import argparse
import yaml
from pathlib import Path


# 预期的保守配置值
EXPECTED_CONSERVATIVE_CONFIG = {
    'actor_rollout_ref.actor.kl_loss_coef': 0.1,
    'actor_rollout_ref.actor.optim.lr': 1e-5,
    'actor_rollout_ref.actor.grad_clip': 0.5,
    'actor_rollout_ref.actor.clip_ratio': 0.1,
    'actor_rollout_ref.model.moe.balance_weight': 0.2,
    'actor_rollout_ref.model.moe.z_loss_weight': 0.01,
    'actor_rollout_ref.model.moe.top_k': 2,
    'trainer.total_epochs': 10,
}


def get_nested_value(d, path):
    """从嵌套字典中获取值，path 用点分隔"""
    keys = path.split('.')
    value = d
    for key in keys:
        if isinstance(value, dict):
            value = value.get(key)
        else:
            return None
    return value


def validate_config(config_path):
    """验证配置文件"""
    print(f"验证配置文件: {config_path}")
    print("=" * 60)

    # 读取配置文件
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 验证关键配置
    all_ok = True
    for path, expected_value in EXPECTED_CONSERVATIVE_CONFIG.items():
        actual_value = get_nested_value(config, path)

        if actual_value is None:
            print(f"❌ {path}: 未找到")
            all_ok = False
        elif actual_value != expected_value:
            print(f"❌ {path}:")
            print(f"   期望: {expected_value}")
            print(f"   实际: {actual_value}")
            all_ok = False
        else:
            print(f"✅ {path}: {actual_value}")

    print("=" * 60)

    # 打印 MoE 配置摘要
    if 'actor_rollout_ref' in config and 'model' in config['actor_rollout_ref']:
        moe_config = config['actor_rollout_ref']['model'].get('moe', {})
        if moe_config.get('enabled', False):
            print("\n🔧 MoE 配置:")
            print(f"  - 专家数量: {moe_config.get('num_experts')}")
            print(f"  - Top-K: {moe_config.get('top_k')}")
            print(f"  - Expert LoRA Rank: {moe_config.get('expert_lora_r')}")
            print(f"  - Balance Weight: {moe_config.get('balance_weight')}")
            print(f"  - Z-Loss Weight: {moe_config.get('z_loss_weight')}")

    # 打印训练配置摘要
    if 'actor_rollout_ref' in config and 'actor' in config['actor_rollout_ref']:
        actor_config = config['actor_rollout_ref']['actor']
        print("\n⚡ 训练配置:")
        print(f"  - 学习率: {actor_config.get('optim', {}).get('lr')}")
        print(f"  - KL Loss Coef: {actor_config.get('kl_loss_coef')}")
        print(f"  - Grad Clip: {actor_config.get('grad_clip')}")
        print(f"  - Clip Ratio: {actor_config.get('clip_ratio')}")

    if 'trainer' in config:
        trainer_config = config['trainer']
        print("\n📊 Trainer 配置:")
        print(f"  - 总轮数: {trainer_config.get('total_epochs')}")
        print(f"  - 项目名: {trainer_config.get('project_name')}")
        print(f"  - 实验名: {trainer_config.get('experiment_name')}")

    print("=" * 60)

    if all_ok:
        print("\n✅ 所有配置验证通过！")
        return 0
    else:
        print("\n❌ 配置验证失败，请检查上述错误")
        return 1


def main():
    parser = argparse.ArgumentParser(description='验证 MoE 训练配置')
    parser.add_argument(
        '--config',
        type=str,
        default='examples/qwen_gui_moe/config/traj_grpo_moe.yaml',
        help='配置文件路径'
    )

    args = parser.parse_args()

    # 解析相对路径
    if not os.path.isabs(args.config):
        # 假设从项目根目录运行
        script_dir = Path(__file__).parent.parent.parent.parent
        config_path = script_dir / args.config
    else:
        config_path = Path(args.config)

    if not config_path.exists():
        print(f"❌ 配置文件不存在: {config_path}")
        return 1

    return validate_config(config_path)


if __name__ == '__main__':
    sys.exit(main())
