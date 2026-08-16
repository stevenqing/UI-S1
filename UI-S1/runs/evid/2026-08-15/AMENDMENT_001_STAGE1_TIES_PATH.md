# Amendment 001: Stage 1 rho-grid ties and path criterion

Date: 2026-08-15

Status: `FROZEN_AFTER_STAGE0_BEFORE_STAGE1_IMPLEMENTATION_OR_RESULT`

Stage 0 passed E-G1/E-G2 and failed E-G3. This amendment does not change those gates or authorize Stage 2.

For secondary variant R, enumerate the Cartesian rho grid in ascending `rho_v`, then ascending `rho_l`. Select maximum inner-validation accuracy; exact ties choose the earliest enumerated pair. E-K5 triggers if either selected coordinate is 0 or 1. The grid is not expanded.

For the diagonal sensitivity path, define cumulative disagreement $d(t)$ as the fraction of rows whose selected block at $(t,t)$ differs from the rho-zero B3 block. The transition is systematic only when Spearman correlation between `t` and `d(t)` over `t=0,0.1,...,1` is strictly greater than 0.8 and `d(1)>d(0.1)`. Otherwise E-K4 triggers and only the path-unification narrative is removed.

For E-K3, “distinguishable” means the 99% paired application-bootstrap CI versus nested dev-selection has lower bound greater than zero. E-K3 triggers when fixed finite EVID is distinguishishable by this definition but the exact-singleton control is not.