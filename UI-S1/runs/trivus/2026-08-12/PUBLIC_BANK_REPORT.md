# TriVUS AndroidControl Public Bank Report

Date: 2026-08-12

Status: `PASS_TRIVUS_PUBLIC_BANK_LOCKED`

- records: 4,000, with 2,000 Low and 2,000 High;
- candidates: three source-neutral actions per record;
- public JSONL SHA-256: `4e2de8d33ab45a0cd7acb33dad8db26dfbca35d8e71851cb8c3b8b7c99aaf7dd`;
- ordered sample-key SHA-256: `d8d6580c3e57c061963bae4ec1d14b02fc82248f79bfdfa33778cc38dbd0487f`;
- actual extracted PNG files: 4,000; unique hashes: 1,988; Low/High paired hash mismatches: zero;
- public candidate-order permutation counts: 669 / 674 / 677 / 677 / 677 / 626 across the six possible orders;
- full schema/image/hash audit: PASS;
- private labels created: false;
- GT fields used: false;
- scorer/evaluator imported: false;
- label-derived metric computed: false.

The public JSONL is retained outside Git because run JSONL files are ignored by repository policy. `data/PUBLIC_MANIFEST.json` is the committed authority. Selector inference is authorized only from this exact public hash.