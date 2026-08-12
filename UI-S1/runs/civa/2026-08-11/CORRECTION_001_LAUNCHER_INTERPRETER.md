# Correction 001: Preserve Active Virtual Environment

Date: 2026-08-11

Timing: after the first launcher invocation and before any CIVA data import, private-label access, model fit, pretest record, outer result, or adjudication.

All five initial workers failed at `import numpy` because `Path(sys.executable).resolve()` followed the `.venv-scaleup/bin/python` symlink to a system interpreter without project packages. The failure occurred at line 8 of `civa_train.py`; no worker reached `load_inputs` or `load_label_folds`, and the failed output directory contained logs only.

The five retained logs are byte-identical, 195 bytes each, with SHA-256 `b56ddc008eb67f0e3a066c5617e46c010ba9e5cfe666b4fa8e9a661c3132de28`. They are stored under `invalid_runtime_001/logs/`.

The launcher now passes `sys.executable` without resolving symlinks. A unit test requires the child interpreter path to equal the active parent interpreter path exactly. This correction changes only environment routing; features, labels, policies, learner, seeds, thresholds, controls, gates, and statistics are unchanged.