# UI-S1 Bridge 缺失资产清单与恢复指引

## 结论

这些研究产物受 `.gitignore` 影响，GitHub clone/pull不会包含。当前源工作区已将它们打成一个经过完整恢复测试的迁移包：

| 项目 | 值 |
|---|---|
| 资产包 | `outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz` |
| 包内文件 | 62 |
| 原始总字节 | 130,025,816 bytes（约130.0 MB / 124.0 MiB） |
| 压缩包 | 9,201,720 bytes（约9.20 MB / 8.78 MiB） |
| 压缩包 SHA256 | `04f97f020d264f65121d97c42088af6181e83f670a337f2930c234559f41ab8f` |
| 外部manifest | `outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.manifest.json` |
| SHA256SUMS | `outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.SHA256SUMS` |

资产包已验证：

- 重复构建SHA256完全一致；
- 源目录verify-only：62/62 kept；
- 空临时目录完整恢复：62/62 restored；
- 没有缺失或checksum失败。

---

## A. 必须从原工作区复制：Pass@8上游源文件

| 相对路径 | Bytes | SHA256 | 用途 |
|---|---:|---|---|
| `outputs/rl_feasibility/per_step.jsonl` | 28,024,817 | `71ab0df74d5f25a8aba5a77cba15959ca3ae390b6dcc979d4178dc4d67e22cfc` | frozen student候选、baseline与critical-step诊断 |
| `outputs/multiagent_complementarity/target_ids.json` | 265,632 | `509327c7de49565e423afd9d8078631fe1df02a5c2af1440c2682e59870f71ad` | 962个Pass@8 target IDs |
| `outputs/multiagent_complementarity/qwen3_vl_candidates.jsonl` | 8,404,978 | `c3e4e0f6984ac5a106b5dc585ce584e1046e033a08ed4d37086df5fd9e63a6ca` | Qwen3-VL K=8 proposals |
| `outputs/multiagent_complementarity/qwen35_candidates.jsonl` | 9,218,496 | `aaf05b9cc36c081b69f48abcc4b6b7c26a1648fa2ca12fe7f84a40070d4fe08f` | Qwen3.5 K=8 proposals |
| `outputs/multiagent_complementarity/extra_tiers/llava15_candidates.jsonl` | 8,152,084 | `fa168f868559467eca220d49b11eebe9559bac6e64a76e4a39b3299d0a7f005e` | LLaVA K=8 proposals |

合计54,066,007 bytes（约54.1 MB）。

---

## B. 建议整体替换：Pass@8完整冻结目录

整体路径：

```text
outputs/pass8_selector_study/
```

源目录共有51个文件、13,664,904 bytes。建议整体恢复，而不是人工挑文件。核心内容包括：

```text
frozen_v1/manifest.json
frozen_v1/screenshot_namespace.json
frozen_v1/blind/{smoke,dev,locked_test}.jsonl
frozen_v1/sealed_labels/{smoke,dev,locked_test}.jsonl
runtime/selectors/{current,strong,exact_plurality,cross_source_consensus}/
eval/{smoke,dev,locked_test}/
bridge_diagnostics/locked_v1/
```

用途：

- 冻结split与blind packet；
- sealed GT/provenance；
- 9B/35B/consensus/plurality输出；
- per-step utility；
- self-source与training-purity bridge诊断。

完整文件级hash位于包内 `_migration_bundle/asset_manifest.json` 和 `_migration_bundle/SHA256SUMS`。

---

## C. 必须从原工作区复制：Purity Bridge / Oracle Replay

| 相对路径 | Bytes | SHA256 | 用途 |
|---|---:|---|---|
| `outputs/multiagent_trajectory_revision/full_v1/causal_arms/a1_gt_target_gt_history.jsonl` | 26,946,055 | `03865e32e0bac2b59177d7cf656bacba3837b5666655ec8078e1401fac2db046` | GT clean replay pool |
| `outputs/multiagent_trajectory_revision/full_v1/causal_arms/a5_revision_target_gt_history.jsonl` | 27,726,919 | `f10bc370acd8997d830cd944497780d48f939d72e699a6f1b412c5a1a7fd8df3` | revision/hard-negative pool |
| `outputs/multiagent_trajectory_revision/full_v1/causal_eval/a5_gt_history_grid/merged.jsonl` | 1,828,270 | `9e286d50434f68bb7c4a4520f898608a8ee62f9ea5d03a31f7b858f9f48421fd` | starting-student correctness |
| `outputs/multiagent_trajectory_revision/full_v1/utility_gate/a13_oracle_student_rescue_gt_history.jsonl` | 1,668,375 | `fb0e892e6df3d7edc894741dffac7ca3c85007ca9d314046d1b9bfa405352001` | 800-row pure rescue bank |
| `outputs/multiagent_trajectory_revision/full_v1/utility_gate/a15_student_rescue25_replay75.jsonl` | 1,698,151 | `7f47ec0eb2a9ae37d01a8af324242cf60aa391a367be1debdadfc427c220242b` | 已验证25/75 anchor |

合计59,867,770 bytes（约59.9 MB）。

---

## D. Train manifest：包内安全副本

相对路径：

```text
outputs/validation_2k/data/train_episodes.jsonl
```

| 属性 | 值 |
|---|---|
| Bytes | 2,496,767 |
| 正确SHA256 | `7af451fb32cd3df60c19a3f281c4b59cb574300519d0bb7d5e961d0bf9d6958e` |
| Episodes | 1,573 |

注意：对正确manifest直接执行Python `random.Random(42).shuffle(lines)` 会得到另一个hash：

```text
004e98c5493b8b5c2ad12641dc7e24e18c34fc502618aa1840f4832a801c4a4d
```

因此不要把“seed 42 shuffle”当成普遍修复。只有当现有文件明确是上述seed-42置乱结果时，才能按该置换做逆变换。迁移包已经包含正确manifest，推荐直接恢复；若目标文件hash已正确，恢复脚本会保留它。

---

## E. 已齐全、无需放入本资产包

根据当前bridge verification，以下资产已经完整；它们体积大或可从固定Hub revision重建，因此未重复打包：

- `.venv-qwen3-vllm` 与 `.venv-qwen35-vllm`；
- 四个Bridge checkpoint：GUI-360 student、Qwen3-VL-8B、Qwen3.5-9B、LLaVA-1.5-7B；
- GUI-360 balanced parquet；
- train/test共20,072张PNG；
- `test_episodes.jsonl`；
- checkpoint与环境的完整安装方法见迁移总指南。

环境和checkpoint仍不应跨机器复制虚拟环境目录；在新机器按固定requirements/Hub revision重建。

---

## F. 复制资产包到目标代码

从源机器发送一个9.2 MB文件即可：

```bash
rsync -a --partial --info=progress2 \
  OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz \
  ./outputs/migration_bundle/
```

或：

```bash
scp OLD_HOST:/home/aiscuser/UI-S1/UI-S1/outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz \
  ./outputs/migration_bundle/
```

校验压缩包：

```bash
echo '04f97f020d264f65121d97c42088af6181e83f670a337f2930c234559f41ab8f  outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz' \
  | sha256sum -c -
```

---

## G. 安全恢复

先只检查当前目标目录：

```bash
.venv-qwen35-vllm/bin/python scripts/restore_migration_asset_bundle.py \
  --bundle outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz \
  --root . \
  --verify-only
```

若目标文件缺失，执行恢复：

```bash
.venv-qwen35-vllm/bin/python scripts/restore_migration_asset_bundle.py \
  --bundle outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz \
  --root .
```

默认行为：

- hash已正确：保留；
- 文件缺失：恢复；
- 文件存在但hash冲突：拒绝覆盖并在report列出。

确认冲突文件确实应由原冻结版本替换后，再显式覆盖：

```bash
.venv-qwen35-vllm/bin/python scripts/restore_migration_asset_bundle.py \
  --bundle outputs/migration_bundle/ui_s1_bridge_missing_assets_v1.tar.gz \
  --root . \
  --overwrite
```

恢复报告默认写入：

```text
outputs/migration_verification/restored_assets.json
```

---

## H. 恢复后总验证

```bash
.venv-qwen35-vllm/bin/python scripts/verify_migration_setup.py \
  --profile bridge \
  --output outputs/migration_verification/bridge_after_restore.json
```

必须满足：

```text
ok=true
failures=[]
```

该验证同时检查：

- 两个uv环境；
- 四个Bridge checkpoint；
- 13个核心文件hash；
- Pass@8 blind/sealed/selector/per-step路径；
- train/test PNG数量。

---

## I. 当前包的生成与重建

重新生成同一资产包：

```bash
.venv-qwen35-vllm/bin/python scripts/build_migration_asset_bundle.py \
  --output-dir outputs/migration_bundle \
  --name ui_s1_bridge_missing_assets_v1
```

构建是确定性的：源文件不变时，重复构建应产生同一SHA256。

完整环境、checkpoint与大数据集迁移说明见：

```text
docs/migration_checkpoints_env_data_zh.md
```
