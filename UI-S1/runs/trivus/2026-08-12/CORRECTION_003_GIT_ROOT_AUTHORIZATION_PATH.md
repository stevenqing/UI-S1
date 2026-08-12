# Correction 003: Git-Root Authorization Path

Date: 2026-08-12

Timing: after result-free implementation commit `d039d31af06090056023bb821b657109bd835e2a` and private-scale authorization commit `975fc78791757befd8722bcb51da71899dcd7080`, before any private-scale record, authorization receipt, fallback context, metric, or model fit.

The first private-scale invocation failed inside committed-authorization validation. The VS Code workspace root is `/home/aiscuser/UI-S1/UI-S1`, while the Git repository root is `/home/aiscuser/UI-S1`. The validator incorrectly addressed the committed blob as `runs/...`; the actual Git path is `UI-S1/runs/...`.

The failure occurred before protected/private target-scale construction and before authorization consumption. No private-scale output directory or authorization receipt was created. PID 2274 was not altered.

The correction resolves Git paths relative to `git rev-parse --show-toplevel`, requires tracked paths to remain inside that root, and adds a regression test that reads the exact committed authorization blob and compares its SHA-256 to working-tree bytes. Untracked authorization files remain rejected.

Because the implementation hash changed, authorization `975fc78` is stale and cannot execute. A new result-free implementation commit and a new one-time authorization commit with a fresh nonce are required before retrying private-scale sealing.