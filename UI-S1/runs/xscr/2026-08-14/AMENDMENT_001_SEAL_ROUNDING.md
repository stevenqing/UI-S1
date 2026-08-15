# Amendment 001: screen-seal rounding

Date: 2026-08-14

Status: `FROZEN_BEFORE_SCREEN_SEAL_AND_ANY_XSCR_STATISTIC`

The specification's `round(0.30 * n_screens)` is fixed to decimal round-half-up. For a stratum with more than one screen, the selected count is additionally capped at `n_screens - 1` so at least one exploratory screen remains. No screen assignment or XSCR statistic existed when this amendment was written.