# XSCR: same-screen cross-row structure feasibility

Round: `xscr`

Date: 2026-08-14

Status: `POST_SELECTION_FEASIBILITY_PREREGISTERED_BEFORE_SCREEN_SEAL_AND_STATISTICS`

GPU: zero. All inputs are frozen banks and retained labels.

## Evidence status and scope

XSCR is a descriptive feasibility check, not a method evaluation and not a paper result. Mind2Web and AndroidControl have already been used in prior label-dependent analyses. Consequently, no subset of their current rows can become untouched confirmation data now.

Before any Q1-Q4 statistic, XSCR nevertheless seals approximately 30% of byte-identical screens as a `PROSPECTIVE_INTERNAL_HOLDOUT`. This prevents the eventual method design from seeing those rows, but evaluation on that holdout remains post-selection internal validation. A confirmatory method claim still requires genuinely new untouched data.

Mind2Web and AndroidControl are co-primary descriptive lanes. They are adjudicated separately because their frozen pools contain 12 and 3 candidates respectively. No pooled effect or cross-benchmark denominator is allowed. Existing project statuses do not change.

## Frozen banks and units

### Mind2Web

- Unit: one base trajectory-step row in the frozen `C_uni` arm.
- Scope: all 2,080 `C_uni` rows. Coordinate-based quantities naturally evaluate false when the selected prediction has no coordinate.
- Public bank: `runs/visual-utility-selector/2026-08-11/data/public_records.jsonl`, filtered to `sample_key` prefix `mind2web/C_uni/`.
- Private candidate labels: the five `private_labels_fold-*.jsonl` files in the same data directory.
- Screen key: public `image_sha256`.
- Split stratum: frozen outer `fold`. This is used instead of opening source task labels to recover website before the seal.
- Candidate order: the public bank order, fixed before labels.
- Coordinate space: prediction coordinates are transformed to image-diagonal coordinates exactly as in GRAN, using the frozen image bytes and public `image_path`.

### AndroidControl

- Unit: one row within one frozen setting. Low and High are evaluated separately and reported side by side.
- Scope: 2,000 Low and 2,000 High rows from the TriVUS public bank.
- Public bank: `runs/trivus/2026-08-12/data/public_records.jsonl`.
- Private candidate labels: the five `runs/trivus/2026-08-12/data/private_labels_fold-*.jsonl` files.
- Screen key: public `image_sha256`. The Low/High pair for the same image is assigned to the same holdout side.
- Split stratum: frozen outer `fold`; paired Low/High screens must have the same fold.
- Candidate order: the public seeded candidate order, fixed before labels.
- Coordinate space: normalized AndroidControl coordinates, matching the frozen TriVUS scorer.

## Screen seal

Seed: `20260814`.

The seal builder may read only the two public banks and their manifests. It must not open private labels, target boxes, source reference rows, correctness fields, or result files.

For each benchmark and stratum, screens are ranked by SHA-256 of `seed | benchmark | stratum | image_sha256`. The first `round(0.30 * n_screens)` are sealed, with at least one exploratory screen retained whenever a stratum has more than one screen. AndroidControl selection operates on the union of Low/High image hashes, then applies the same assignment to both settings.

The seal manifest records every screen key and side, input hashes, counts by stratum, seed, and its own SHA-256. Q1-Q4 scripts must refuse to run unless this manifest exists and matches the committed public inputs. They must exclude sealed screens before opening any label-dependent input.

## Label-free top mode

For a tolerance $\tau$, parsed candidates are partitioned by deterministic complete-link clustering. Two candidates are equivalent only when their actions match and either:

- both have coordinates within Euclidean distance $\tau$ in the lane's frozen coordinate space; or
- both lack coordinates and their parameters match exactly.

The highest-weight mode is the class with most members. Ties are resolved by the smallest original candidate index. The representative is the smallest-index member of the winning class. The mode weight is its unweighted member count.

No fold-local reliability or correctness-derived weight enters Q1 or Q2.

## Tolerance grids

GRAN did not produce one global numeric Mind2Web $\tau^*$. Its nested selections contain two finite values, `0.0022908676527677724` and `0.50118723362727224`, plus the non-geometric `single` option. XSCR therefore reports the sorted union of half, original, and double for each finite value:

`[0.0011454338263838862, 0.0022908676527677724, 0.004581735305535545, 0.2505936168136361, 0.5011872336272722, 1.0023744672545445]`.

The `single` option is reported as an inadmissible spatial boundary and is not converted into a distance.

AndroidControl has no GRAN sweep. It uses the previously frozen scorer radius and its half/double sensitivity: `[0.07, 0.14, 0.28]`.

No tolerance is selected inside XSCR.

## Quantities

All quantities use exploratory screens only.

### Q1: structure existence

For each lane, report rows per screen: row count, screen count, median, linear-interpolation quartiles, singleton-screen fraction, and fraction of rows on singleton screens. Q1 uses no labels.

### Q2: collision rate

At every tolerance, a row collides when its selected representative has a coordinate and at least one other row on the same screen has a selected representative within $\tau$, regardless of correctness. Report collision rows divided by all exploratory rows, plus the count of screens containing a collision. Q2 uses no labels.

### Q3: optimistic repair surface

A row is recoverable when its selected representative is wrong and at least one candidate is correct. It is structurally repairable when another same-screen row has a colliding selected representative with strictly larger mode weight. Equal weights do not imply a forced winner.

Report the repairable count, repairable/recoverable, and repairable/all-exploratory-rows. This is an optimistic structural upper surface; it does not assert that the displaced row selects a correct second mode.

### Q4: damage surface

A selected-correct row is structurally damageable when another same-screen row has a colliding selected representative with strictly larger mode weight. Report the damageable count, damageable/selected-correct, and damageable/all-exploratory-rows.

Q3 and Q4 must appear together for every tolerance. The signed screening proxy is:

$$
100\frac{n_{repairable}-n_{damageable}}{N_{exploratory}}\ \text{percentage points}.
$$

Because both terms are structural bounds, this proxy is a design-screening quantity rather than a guaranteed method gain. Compare it descriptively with the 0.70 pp MDE.

### Shared-target diagnostic

After Q3/Q4 label access is authorized, report among coordinate-target rows on multi-row screens the fraction having another same-screen ground-truth location within each tolerance. Mind2Web uses the target-box center in image-diagonal coordinates; AndroidControl uses its frozen normalized ground-truth coordinate. This is evaluation-side evidence for soft rather than hard exclusion.

## Sequential execution and human decision

1. Commit this specification and implementation-free configuration.
2. Build and commit the public-only screen seal and input manifest.
3. Compute and commit Q1 only.
4. Record a human `PROCEED` or `STOP` decision with a reason. If stopped, do not compute Q2-Q4.
5. If authorized, compute and commit Q2 only, then record a second human decision.
6. If authorized again, open private labels and compute Q3/Q4 together.
7. Write the report, retention manifest, and status.

The human should stop when singleton screens dominate, collisions are negligible, or the signed full-lane proxy is consistently below 0.70 pp. These are guidance, not automatic gates.

## Discipline

Q3, Q4, shared-target statistics, and all correctness-derived values are evaluation-side only and cannot define a runtime rule. The five leaked ScreenSpot-Pro cells are irrelevant and prohibited. Any later joint-assignment method is transductive, changes the evaluation protocol by requiring all same-screen rows together, must use soft constraints selected on development data, and must compare against majority, nested dev-selection, and a random-assignment control. It cannot be mixed into the existing main table as a same-setting improvement.

Derived row JSONL must be written with write/flush/fsync. Every source, generated artifact, and dataset snapshot is SHA-256 locked. Raw derived rows stay Git-ignored and are copied to an independent scratch retention path; manifests, decisions, reports, and status are committed.