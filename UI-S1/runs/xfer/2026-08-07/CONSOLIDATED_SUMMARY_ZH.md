# Cross-Benchmark Q1 Transfer 总结

日期：2026-08-07

状态：`MIND2WEB_PASS`

## Mind2Web 主结果

| Arm | Micro Step SR | Episode macro Step SR | Mean forwards |
|---|---:|---:|---:|
| C_uni | 26.68% | 32.26% | 12.000 |
| C_cond | 31.59% | 37.37% | 12.000 |
| C_rand | 28.32% | 33.28% | 12.000 |
| C_self | 29.28% | 34.49% | 12.000 |

## 配对比较

- C-cond − C_uni: +4.90 pp, 99% CI [+2.94, +6.86] pp.
- C-cond − C_rand: +3.27 pp, 99% CI [+1.26, +5.32] pp.
- C-cond − C_self: +2.31 pp, 99% CI [+0.95, +3.68] pp.

## 预注册裁决

- XF1: **True**；Mind2Web MDE = +0.61 pp.
- XF2: **True**.
- XF4: **False**.
- XF-K1: **False**.
- XF-K2: **False**.
- XF-K3: **False**；stage-2 trigger rate = 100.00%.
- AndroidControl decision: **PROCEED**.

## 必报诊断

- Triggered C-cond Step SR: 31.59%.
- Non-triggered C-cond Step SR: N/A.
- Rank-0 full-bbox containment: 40.38%.
- Mean rank0–11 full-bbox containment: 35.12%.
- Single-cluster geometry fallback rows: 37.
- Max arm mean-forward difference: 0.000; budget-matched control required: False.

## 产物

- `mde_mind2web.json`
- `xf_mind2web.json`
- `STATUS.json`
- `raw/mind2web-consensus-roi.jsonl`
- `/scratch/workspaceblobstore/xfer-traces/2026-08-07/BACKUP_MANIFEST.json`
