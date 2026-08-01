# Amendment 007: H3 Shared View Regions

Date: 2026-08-01

Status: frozen after H1 and before H3 mixed-pool inference.

H3 separates pool diversity from proposal diversity by sharing view geometry across models. For each ScreenSpot-Pro identity, use the GTA1 H1 N=4 candidate regions exactly:

1. full image;
2. GTA1 official attention subimage 1 region;
3. GTA1 official attention subimage 2 region;
4. GTA1 official attention subimage 3 region.

Qwen3-VL-8B and the third eligible model run greedy localization on these same full/crop images. They do not generate their own attention regions. Crop coordinates are mapped back to original-image pixels using each model's released processor contract. No target bbox enters crop selection or inference.

D1 contains GTA1 candidates 0-11 from the H1 superset. D2 contains four candidates from each of three eligible lineages under the shared region geometry above. All candidates retain `(model, view_index, region)` metadata. B3 and M1 receive exactly the same 12-candidate tensor per D1/D2 row.

Qwen3-VL-8B eligibility is established before H3 from an existing independent trace. Exact key matching covers all 1,581 pinned identities; fail-closed full-image accuracy is 864/1,581 = 54.65%, above the frozen 24.70% threshold. Extra duplicate exports are excluded by exact `(filename,bbox,instruction)` matching. The H3 four-view rows are regenerated from the pinned official checkpoint revision `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b`.

The third model remains unset until its independent full-image ScreenSpot-Pro score is complete. If no third local checkpoint exceeds 24.70%, H3 is blocked rather than changing the three-lineage requirement.