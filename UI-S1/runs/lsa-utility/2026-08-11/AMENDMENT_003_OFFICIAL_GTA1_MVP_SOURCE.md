# Amendment 003 — Official GTA1 / MVP Source Semantics

Date: 2026-08-11

Status: `PRE_RESULT`

Official-source verification before any Utility-LSA result established:

- GTA1 repository: `Yan98/GTA1`, current source inspected in `src/grpo_grounding.py`, `src/trainer/grpo_trainer.py`, and `src/trainer/grpo_config.py`.
- GTA1 uses `num_generations=8`, click-in-target binary accuracy reward, and group-relative advantage `(reward - mean)/(std + 1e-4)`. The reconstructed spec previously wrote `1e-5` from a local UI-R1-derived trainer; this amendment corrects Utility-LSA to official GTA1's `1e-4`.
- GTA1 trains the grounding policy. It does not train the MVP aggregation rule.
- MVP pinned source: `ZJUSCL/MVP@988ff3c61b9f7632d780ae27c83260de75b3c95f`.
- MVP aggregation uses deterministic 14-pixel complete-link grouping, cluster size as the primary score, mean AGVP coverage as tie-break, and returns the highest-coverage real member.

No Utility-LSA target, feature, model grid, threshold grid, or gate is otherwise changed.
