# TriVUS AndroidControl Blind Selector Report

Date: 2026-08-12

Status: `PASS_TRIVUS_SELECTOR_BLIND_LOCK`

- public records: 4,000, Low/High 2,000 each;
- blind predictions: 4,000, Low/High 2,000 each;
- public SHA-256: `4e2de8d33ab45a0cd7acb33dad8db26dfbca35d8e71851cb8c3b8b7c99aaf7dd`;
- prediction SHA-256: `b3907c5350b9b981eec15445e2caa8ea82dd2ea40474242334b2a035b20e178d`;
- model-index SHA-256: `520b2e05079402e9468a8701d03d1154d14b2599593afb6effa7fb60c1bff070`;
- eight shards: exactly 500 unique rows each, exact disjoint union coverage;
- prompt hashes: reconstructed and verified for all 4,000 rows;
- rendered overlay hashes: reconstructed and verified for all 4,000 rows;
- selected labels: independently verified as A--C probability argmax with correct candidate permutation mapping;
- finite logits/probabilities and probability sums: PASS;
- private labels created: false;
- GT fields used: false;
- scorer/evaluator imported: false;
- label-derived metrics computed: false.

The merged JSONL and shard JSONLs remain ignored by Git and are retained in the authoritative external per-file SHA-256 snapshot. `selector/BLIND_MANIFEST.json` is the committed evidence lock.

Only after this checkpoint is published may the physically fold-sealed private-label builder access AndroidControl GT fields. No selector accuracy, AUROC, model training, or threshold selection has occurred at this checkpoint.