# CWIN Containment Reconciliation

Date: 2026-08-17

Status: `PASS_P0_BEFORE_ANY_CWIN_GEOMETRY_OR_RESULT`

No CWIN L1-L4 quantity or window was computed during this reconciliation.

## The apparent contradiction

The historical ScreenSpot-Pro containment curve begins at 99.94% full-bbox containment and ends at 61.04%. COVER reports 225 rows (14.23%) whose target center is outside the union of the existing crop windows. These values use different window sets.

The frozen N12 region manifest defines:

- view 0: `[0,0,width,height]`, the full image;
- views 1-11: GTA1 attention-ranked crop regions.

Allocation-Law rank 0 is therefore the full-image view. Its 99.94% full-bbox containment is near-trivial; the small deficit comes from annotation/image-bound edge semantics. Rank 11 is the final individual crop. COVER deliberately excludes view 0 and computes the union of views 1-11, so target-center uncovered rows are possible. There is no numerical contradiction.

Authoritative sources:

- `runs/allocation-law/2026-08-01/raw/shared_regions_n12.jsonl`, SHA-256 `2a7233cf0fbab109ea481ba891d2d7e65a481c48cd73c623d7d76fc61c06ae17`;
- `runs/allocation-law/2026-08-01/L4_RESULTS.json`, rank-0 full-bbox containment `0.9993674889310563`, rank-11 `0.6103731815306768`;
- `runs/cover/2026-08-16/ARM_A.json`, crop-only uncovered target-center fraction `0.14231499051233396`.

## E3 wording

E3's numerical statement remains correct: the SSPro proposal sequence starts near full containment and decays strongly by late rank, while Mind2Web starts lower and decays less. The mechanism wording requires containment:

> SSPro's high start is the full-image baseline, not evidence that the first adaptive crop is exceptionally targeted. E3 measures full-image-to-late-crop proposal-quality decay. Its high-start condition remains a qualitative boundary, not a zoom-quality law.

No existing E3 result or status changes.

## Budget reconciliation

The 11 crop geometries are shared across model lineages, but forward budgets are model-view pairs.

- Stage 1a is GTA1 V-only N12: one full-image forward plus 11 crop forwards. Replacing $K$ crops with $K$ complementary crops preserves 12 GTA1 forwards.
- Stage 1b, if separately authorized, is the complete three-lineage 36-forward bank: three full-image forwards plus 33 crop forwards. The same $K$ geometries are replaced independently for each lineage, preserving 36 forwards and requiring `2K*1581` additional forwards beyond Stage 1a.

The mixed C-uni N12 pool has only views 0-3 per lineage and is not the Stage-1 replacement pool. It remains available only as a mandatory historical baseline. This prevents mixing 11 geometric crop ranks with a three-crop-per-lineage budget.

## Consequence

CWIN is permitted to implement zero-GPU geometry and a strict oracle net-gain upper bound. GPU remains unauthorized until Stage 0 passes and a new amendment freezes all execution details.