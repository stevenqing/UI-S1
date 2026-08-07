# Cross-Benchmark Transfer Spec (Q1)

Date: 2026-08-07

Status: result-blind execution freeze.

Upstream: `runs/consolidate/2026-08-06/`, where ScreenSpot-Pro Q1 C-cond reaches B3 65.91% versus C-uni 63.69%, delta +2.21 pp with 99% CI [+0.50,+4.16], and rank decay is the primary observed decline mechanism.

## Scope

Transfer the four-arm Q1 design to Mind2Web and AndroidControl. Historical row traces are missing, so only the minimum three-lineage roster is regenerated. Old aggregate scores remain historical context and never enter paired differences with new traces.

## Retention gate

Before inference:

- every row prediction is JSONL, flushed and fsynced per row;
- each completed lane receives a SHA-256 manifest record;
- each completed lane is copied to `/scratch/workspaceblobstore/xfer-traces/2026-08-07/` and hash-verified;
- local `raw/` and `predictions*.jsonl` are protected from recursive cleanup;
- `STATUS.json` must record local and backup paths.

## Rosters

Mind2Web:

- TongUI-7B: proposer and scorer;
- CogAgent-18B: scorer;
- UI-TARS-7B: scorer.

AndroidControl Low/High:

- GUI-R1-7B: proposer and scorer;
- UI-AGILE-7B: scorer;
- UI-R1-E-3B: scorer.

Exact repositories, revisions, architecture families, local paths, and the new transfer prompt contract are frozen in `configs/xfer_roster.yaml`.

## Data

Mind2Web uses all 2,080 scoreable Cross-Task actions and reports micro plus 252-episode macro.

AndroidControl excludes 58 parameter-conflict pairs, then samples the same 2,000 paired Low/High identities proportionally by Low GT action with seed 20260807. The frozen IDs and action counts are in `configs/ac_subsample.yaml` and `data/androidcontrol/subsample.jsonl`.

## Product-action Q1

Stage 1 uses six forwards: three lineages by full image and proposer view1. Action type is selected by plurality. Non-coordinate winning types skip stage 2. Coordinate predictions voting for the winning type are clustered; largest and second-largest clusters define two RoIs.

Stage 2 uses two RoIs by three lineages, six forwards when triggered.

All Mind2Web stage-2 model lanes use eight deterministic modulo shards. CogAgent may run shards 4-7 before shards 0-3 when only four GPUs are free; this changes scheduling only, not row assignment, prompts, generation, or aggregation.

Arms:

- C-uni: proposer views 2/3 by three lineages;
- C-cond: cross-lineage consensus RoIs by three lineages;
- C-rand: seeded random crops by three lineages;
- C-self: proposer view0/view1-centered crops by three lineages.

Aggregation uses action plurality, coordinate density grouping, and token-set F1 for parameters. A3/A4 joint PKA is prohibited.

## Reporting and gates

Mandatory per benchmark:

- C-cond versus C-uni, C-rand, and C-self;
- best-single and pool mean;
- second-stage trigger rate;
- triggered and non-triggered subset accuracy;
- mean forward count and a budget-matched control if arm means differ by more than 0.5;
- V-only and Mixed N4/8/12/16 curves;
- proposer rank containment curve;
- benchmark-specific three-seed MDE, defined as twice Step SR SD.

XF1: Mind2Web C-cond minus C-uni exceeds Mind2Web MDE and has positive 99% CI lower bound.

XF2: C-cond minus C-rand and C-self both have positive 99% CI lower bounds.

XF3: AndroidControl applies the XF1 rule separately to Low and High.

XF4: V-only N16-minus-N4 is negative and Mixed is positive, with intervals excluding zero.

Kill conditions:

- XF-K1: Mind2Web C-cond does not beat C-uni; cancel AndroidControl inference and constrain Q1 to ScreenSpot-Pro.
- XF-K2: C-cond beats C-uni but not C-self; weaken the cross-lineage-specific claim.
- XF-K3: stage-2 trigger rate below 60%; constrain the claim to triggered rows.
- XF-K4: AndroidControl subsample single-model bias exceeds 2 pp; move AndroidControl to the appendix.

C-rand, C-self, best-single, and pool mean are mandatory and may not be omitted.
