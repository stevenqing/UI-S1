# Preregistration Amendment 005: Official MVP Semantics

Date: 2026-07-30

Status: applied after locating `ZJUSCL/MVP@988ff3c61b9f7632d780ae27c83260de75b3c95f` and before any W2/W3 result is frozen.

## Source verification

The MVP paper is arXiv `2512.08529`, “MVP: Multiple View Prediction Improves GUI Grounding.” It reports GTA1-7B at 61.7% on ScreenSpot-Pro and links the official repository above.

## v1 naming correction

The W2 `v1` transform remains the preregistered 28-pixel black border on all four sides. Official MVP's GTA1 ScreenSpot-Pro path does not implement this transform as its main view generator; it uses AGVP crops. A separate `get_border_regions` implementation appears in OSWorld/Qwen3 paths with a default width of 32 pixels and is not equivalent to four-sided black padding.

Therefore:

- W2 `v1` is renamed **spec-defined border perturbation**.
- It remains valid for the preregistered K1 early mechanism test.
- It must not be described as an exact MVP Border Padding reproduction.
- W3's exact MVP row uses official AGVP and official aggregation semantics.

## Official aggregation semantics

The official GTA1 ScreenSpot-Pro code:

1. groups points greedily in view order;
2. admits a point only when both absolute x and absolute y differences are at most 14 pixels to every point already in the group (complete-link condition);
3. ranks groups by size, then average AGVP coverage;
4. returns the highest-coverage original prediction in the winning group.

The paper prose describes the centroid of the densest cluster. W3 reports both:

- `MVP_official_code`: exact source behavior, used for the 61.7 sanity anchor;
- `MVP_paper_centroid`: same grouping but centroid output, used as a paper-description sensitivity row.

The earlier algorithm-level connected-component centroid is retained only as `MVP_graph_centroid_ablation` and is not called an official reproduction.

## Official GTA1 parameters

- attention layer: 20;
- target token: `<|box_end|>` by the `mvp_sspro.py` parser default, while `eval_gta1.sh` exports comma; both are tested in the preregistered dev token ablation;
- maximum subimage inferences: 4;
- top attention tokens: 100;
- consistency threshold: 14 pixels;
- subimage resize: 2x before model preprocessing.