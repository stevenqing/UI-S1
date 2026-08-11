# Amendment 005 — GTA1 Sample Standard Deviation

Date: 2026-08-11

Status: `PRE_RESULT`

Official GTA1 computes group reward spread with `torch.std(dim=1)`, whose default is sample standard deviation. Before any Utility-LSA result, U-GRPO/U-HYBRID normalization is corrected from NumPy population standard deviation to `np.std(..., ddof=1)`, retaining the official `1e-4` stabilizer.

No other target, feature, model, threshold, or gate changes.
