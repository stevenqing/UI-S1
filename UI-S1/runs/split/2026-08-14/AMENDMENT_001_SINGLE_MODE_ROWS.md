# SPLIT Amendment 001: Single-mode rows

Date: 2026-08-14
Timing: after preflight commit `6742603`, before any `ZERO_GPU_GATE.json` was written and before any aggregate gate statistic was observed.

## Trigger

The first zero-GPU runner attempt stopped on row `eviews_windows_17` because the inherited finite GRAN threshold produced fewer than two complete-link modes. The preregistration defined $M_1$ and $M_2$ but did not define behavior when $M_2$ does not exist. The failed attempt emitted no result artifact and no aggregate statistic.

## Frozen resolution

A row with fewer than two modes is structurally ineligible for the two-mode ambiguity gate:

- retain the row in the held-out denominator used by $\Delta_2$;
- set `gate=false`;
- set `positive=false` and `negative=false`;
- record the available mode count and increment `insufficient_mode_rows`;
- do not alter $\tau$, synthesize a second mode, or rescue the row with another threshold.

This is the conservative extension of $w_2/w_1\ge g$: when $w_2$ is undefined because no second mode exists, the two-mode probe cannot trigger. All other frozen rules remain unchanged.
