# Amendment 009: Fallback-Context Execution Firewall

Date: 2026-08-12

Timing: after Amendment 008 and representation-gate publication, before any private-scale seal, exact fallback-context artifact, unified feature artifact, or TriVUS model fit.

## 1. Exact nested behavior-policy reference

For outer fold k and OOF holdout h, the checkpoint fold v is the cyclic first member after h among the three remaining outer-development folds. The two remaining folds are model-training folds T.

The VUS-SR behavior-policy call is reproduced exactly as `fit_inner_policies(banks, T, v, cev_config)`. Thus v, not h, is the reference used by the behavior policy to choose its internal CEV configuration-validation fold from T. Candidate reliability and Mind2Web scale still use only T. The resulting fallback is applied to every row in the four outer-development folds.

## 2. Physical private-scale seal

Mind2Web normalized target-box width/height is private behavior-policy state. A separately authorized trusted step converts it into five physical fold files. Each record contains only row ID, normalized width, and normalized height. It emits no target position, success bit, candidate identity, aggregate target statistic, or performance metric.

The five files and their manifest are built in one same-filesystem staging directory and published by one directory rename only after all row counts, hashes, schemas, and protected-process checks pass. Any pre-publication failure leaves no destination directory.

## 3. Exact context transaction

Formal context generation requires a second authorization committed after the private-scale seal. Before any private fold is opened, the authorization nonce is consumed by exclusive creation of a permanent receipt. The same authorization can never be replayed. A failed run requires a newly committed authorization with a new nonce.

Each authorization file must itself be committed unchanged. Its implementation commit must be a strict ancestor of the authorization commit. For formal contexts, the committed private-scale manifest must also be a strict ancestor of the context-authorization commit. Every declared implementation/result hash must match both the working-tree bytes and the exact Git blob in its declared commit.

The 391,524 context rows and their manifest are built in one same-filesystem staging directory. Publication is one directory rename after exact total count, exact per-sample 5-final/16-inner coverage, 14,644 frozen final-index anchors, hashes, schemas, and protected-process identity all pass.

## 4. Path and environment lock

All sealed-label manifest paths must be relative and remain under their declared root. Public image paths must resolve inside the repository and match their frozen hashes. Execution is restricted to `.venv-scaleup/bin/python` with the frozen Python, NumPy, scikit-learn, Torch, and PyYAML versions.

## 5. Output firewall

Context rows contain exactly schema version, context key, sample key, outer fold, role, holdout fold, fit folds, and fallback index. They contain no success bit, target, source/model/lineage/slot identity, reliability, policy configuration, configuration score, or aggregate metric.

Private-scale sealing and fallback-context generation remain preprocessing only. They do not authorize unified data assembly, optimizer steps, checkpoint selection, threshold fitting, or outer evaluation.