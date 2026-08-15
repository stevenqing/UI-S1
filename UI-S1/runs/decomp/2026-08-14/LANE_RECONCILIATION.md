# DECOMP Lane Reconciliation

Date: 2026-08-15

Status: `PASS_P0_RECONCILED_BEFORE_ANY_DECOMP_ARM`

Repository HEAD at reconciliation: `f9739e7460d9ff91a0f5ac9880228bad453fba55`.

No DECOMP arm statistic was computed during this reconciliation.

## XSCR row counts

XSCR's displayed counts are post-seal exploratory rows, not frozen-bank sizes.

| Lane | Full frozen rows | XSCR exploratory rows | XSCR exploratory screens | Source | Source SHA-256 | Producing commit |
| --- | ---: | ---: | ---: | --- | --- | --- |
| Mind2Web `C_uni` | 2,080 | 1,460 | 1,402 | `runs/xscr/2026-08-14/Q1.json` | `7ef9e4348524af4e540182820612b835cfcc332119b1bcaebfccac8fe783a8b2` | `3d435c8554d1596a776c10a31c5f02a824712c21` |
| AndroidControl Low | 2,000 | 1,400 | 1,392 | `runs/xscr/2026-08-14/Q1.json` | same file | same commit |
| AndroidControl High | 2,000 | 1,400 | 1,392 | `runs/xscr/2026-08-14/Q1.json` | same file | same commit |

The screen assignment is locked by `SCREEN_SEAL.json`, SHA-256 `d1b6463074f7c13d562291ea6bbb650c3f3805f242777c9b14633dd45e18dab6`, commit `f7ce483ce7231fcab382d81f0d03cf090a8cfa91`. The public input manifest is SHA-256 `748b0a3498b694c8013c99e1a619da2bb3927ab294fa4985187abc7d6b73a513`, from the same commit.

The Q1 raw file has 4,186 rows because it contains one row per exploratory screen: $1,402+1,392+1,392=4,186$. The displayed lane row counts sum to 4,260 because those count benchmark rows: $1,460+1,400+1,400=4,260$. This is a unit difference, not missing data.

XSCR later read all private-label files. Its nominal holdout is invalid for future independent evaluation, as recorded in `STATUS.json`, SHA-256 `de9060047bad02b21ebbbde74f7a3a695d60a80a0d880b5f72a547cb0bec13a0`, commit `6bc186e5fd6eaf282c7f61684b72c4fc8fd3b40b`.

## SPLIT model names

SPLIT contains two distinct model rosters.

The frozen ScreenSpot-Pro C-uni bank is 12 slots from three lineages and four views:

- `GTA1-7B`
- `Qwen3-VL-8B-Instruct`
- `UI-TARS-7B-SFT`

The proposed falsification-crop probes were separately:

- primary: `Qwen3-VL-8B-Instruct`;
- secondary: `GTA1-7B`;
- deferred: `Qwen2.5-VL-7B-Instruct`, whose checkpoint was missing.

Qwen2.5 was never a C-uni bank lineage. It was a prospective probe checkpoint. SPLIT ran zero model forwards and stopped at geometry. There is no source mismatch once bank lineage and probe role are separated.

Authoritative files:

| File | SHA-256 | Commit |
| --- | --- | --- |
| `runs/split/2026-08-14/configs/split_prereg.yaml` | `578dc10612c38561425d043857fd0b01db8a97380f938d9bb4c04569f2c62736` | `49219327f5a11d55ae241d1aabddd8cafedb794d` |
| `runs/split/2026-08-14/PREFLIGHT.json` | `9188e6e9b886cacbbff52cb43e8223c08a9108750e139ef42d39cede91da2a6b` | `674260385ab133d11e36e11cc94be0c3d92fcc4b` |
| `runs/split/2026-08-14/REPORT.md` | `df22aa20f5627e9e08f93d66c14cfcdaaed654572e9856c6e65ef6010ff35c32` | `0a3b8704ffdef0e36dfb43be090fd818562cb731` |

## ORTH Mind2Web HTML/DOM availability

The official historical lane was completely audited at 2,094 actions from 252 episodes, dataset revision `17ece8eb89862368edc0cc806acee6fca5163474`. The current local workspace lacks both:

- `runs/mindact/2026-07-29/data/Mind2Web`
- `runs/mindact/2026-07-29/data/source/scores_all_data.pkl`

The current 2,080-row XFER data retains 1,975 GT-selected positive snippets, not a complete DOM/AX tree or candidate-element universe. It cannot substitute for the official lane.

| File | SHA-256 | Commit |
| --- | --- | --- |
| `runs/orth/2026-08-14/PREFLIGHT.json` | `8b32da564fe5ebf1bd7d5127d86064bd40a4e8a0ebed240c662c5004edd50358` | `2f3248485063881fa1ba8a2c8940dd890ab3420c` |
| `runs/orth/2026-08-14/ARM2.json` | `c77a56d93aab683c6ff89fe8b4ec9f20f8be8fcdac81c649b1155875158f9941` | `db495889561050b1a0bdcfa5097bb54293576070` |
| `runs/orth/2026-08-14/REPORT.md` | `d5886a176257e9b85ab63ecc5ca020dadf156edf5efd04eba34c4159de342fc5` | `c0b460926dec96eaf713b2031e0993999aac8673` |

## Arm 1 authority and scope

The canonical $+3.6053$ pp result is ScreenSpot-Pro only:

- 1,581 rows;
- mixed C-uni N12: three lineages x four views;
- comparator: GTA1-only V-only N12;
- canonical density B3: 63.6939% versus 60.0886%;
- paired 99% CI: `[+1.310,+6.224]` pp.

The result is recorded in `runs/final/2026-08-04/m0_manifest_diff.json`, SHA-256 `f7fd32ab8da348862430c7926f0073dfd80c71c01595a1bbae9d96fe7b7325fd`, commit `85e2f8151f6f792ec773458a09cd5d1d4c479286`, and `MAIN_TABLE.md`, SHA-256 `7a7bfe33cedd52da0255f02c7bf140a0343bc4930c03603b434c2a1789f5cfb3`, same commit.

Mind2Web does not have a compliant aligned three-lineage x four-view pool. Allocation-Law L3 is `BLOCKED_PREREGISTRATION_GAP`. Its report is SHA-256 `eaeaec310ba77ad7c5766836e617aa91db7bd0c131f2024a0bd34dad9889d726`, commit `959aec122c1b3d1d77fdcaf9d3ae4f355146da83`; its ScreenSpot-Pro L1 result is SHA-256 `b06486a9aa2c1caa026d35d10ee84ae0e67f51080b60f60c26f25f5f51d2fa11`, commit `84e00576941b271ae8782286cc6b4262a48ef3f7`.

Therefore DECOMP Arm 1 is authorized for ScreenSpot-Pro only. Mind2Web is recorded as `BLOCKED_ALIGNED_POOL_UNAVAILABLE` and is not evaluated.

## Arm 2 authority and scale

Arm 2 is ScreenSpot-Pro only and reads no labels. The original target-bbox short-side median $R$ is rejected because target boxes are labels. The replacement grid is the canonical B3 fixed pixel threshold and its half/double sensitivity: `[7,14,28]` pixels. The canonical implementation sets `MVP_THRESHOLD_PIXELS = 14.0` in `runs/ccm-h2h/2026-07-31/h1/aggregators_coord.py`.

The same 1,581-row public image/candidate bank used by the reconciled ScreenSpot-Pro C-uni lane is authorized. Q1 groups by image byte SHA-256 and, when present, `img_filename`; Q2 uses only candidate coordinates and the fixed pixel grid.

## Arm 3 logprob inventory

Neither ScreenSpot-Pro nor Mind2Web retained generating-model per-token logprobs, coordinate-token logprobs, or sequence scores. Existing `label_logits` and `label_probabilities` are downstream selector logits over candidate labels, not probabilities emitted by the generating VLM for coordinate tokens. OCR confidence is also a different channel.

Arm 3 is therefore authorized only for a manifest-backed field inventory and must stop before label access or AUROC. Future generating-model forwards must retain token IDs, token logprobs, sequence score and normalization semantics, coordinate-token span, sampling parameters, model revision, prompt hash, and explicit `logprobs_unavailable` when the backend cannot expose them.

## P0 decisions

| Arm | Decision | Reason |
| --- | --- | --- |
| Arm 1 | `AUTHORIZED_SSPRO_ONLY` | Frozen aligned 3x4 C-uni pool exists only for ScreenSpot-Pro. |
| Arm 2 | `AUTHORIZED_LABEL_FREE_SSPRO` | Uses image hashes, candidate coordinates, and fixed `[7,14,28] px`; no target boxes. |
| Arm 3 | `AUTHORIZED_INVENTORY_ONLY_EXPECT_STOP` | Generating-model logprobs are absent from both banks. |

All other lanes are prohibited from DECOMP computation.