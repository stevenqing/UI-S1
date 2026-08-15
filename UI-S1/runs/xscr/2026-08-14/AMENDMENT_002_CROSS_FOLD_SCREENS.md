# Amendment 002: byte-identical screens crossing folds

Date: 2026-08-14

Status: `FROZEN_AFTER_FAILED_PUBLIC_ONLY_SEAL_BEFORE_ANY_SEAL_OR_XSCR_STATISTIC`

The first public-only seal attempt stopped before writing any artifact because at least one Mind2Web `image_sha256` occurs in rows assigned to different existing outer folds. Therefore the earlier assumption that an identical screen never crosses folds is false for the frozen Mind2Web bank.

Screen integrity takes precedence over fold stratification. A screen's seal stratum is now the sorted comma-separated tuple of all folds in which that byte-identical screen appears, for example `1,3`. Every row sharing the image hash receives one seal side. Low/High AndroidControl screens are handled by the same rule and must expose identical fold sets across settings.

This correction does not inspect labels and does not compute Q1-Q4. A later transductive method round must account for byte-identical screens crossing existing folds; ordinary row-fold evaluation alone is not sufficient isolation for same-screen methods.