# Amendment 006: Fold-Sealed AndroidControl Labels

Date: 2026-08-12

Timing: after the selector blind lock was independently verified and committed as `8b2cc5c549a78cfb54918e5bdd271ebf52387f4f`, before any TriVUS private-label file, selector metric, data adapter fit, or model training.

The private builder must verify that the blind-lock commit is an ancestor of `HEAD`, and verify the committed blind/public manifests plus public/prediction/scorer/fold-map hashes before importing scoring code or indexing GT fields.

It creates five physical files `data/private_labels_fold-k.jsonl`. Each record contains only:

- schema version;
- sample key;
- exactly three candidate-success booleans in the source-neutral public candidate order.

Frozen combined Low+High fold counts are 792 / 754 / 826 / 870 / 758, totaling 4,000. Low and High paired rows remain in the same fold.

Candidate success uses the frozen AndroidControl contract:

- exact action equality;
- normalized Euclidean coordinate distance below 0.14 for grounding actions;
- token-F1 at least 0.5 for type/open_app/scroll/select;
- exact action equality for simple actions;
- unknown, unsupported, or parse-failed candidates are unsuccessful.

The builder writes each fold with flush/fsync, then a manifest containing only identities, row counts, file hashes, schema, and dependency hashes. It must not compute or report success rates, per-model scores, direct selector accuracy, oracle coverage, majority comparisons, AUROC, or any aggregate of the success booleans.

The fold files remain ignored by Git and are retained externally. Only `data/PRIVATE_LABEL_MANIFEST.json` may be committed. Representation-gate code and model training remain unauthorized until that manifest is published and a later result-free amendment freezes the exact diagnostic.