# TriVUS AndroidControl Private-Label Seal Report

Date: 2026-08-12

Status: `PASS_TRIVUS_FOLD_SEALED_PRIVATE_LABELS`

- total records: 4,000;
- candidate-success width: 3 booleans;
- fold rows: 792 / 754 / 826 / 870 / 758;
- fold SHA-256 values:
  - fold 0: `3c0a07f210ad7d7518c40f36b2ff2803fd0452017511614a47345c607d32066a`;
  - fold 1: `d49bcd4f49a924db632a86482a4c0d09cc7b7982846f4a938225aea5e053ff2d`;
  - fold 2: `0a3026f9630cb104fc8c08dc42521fd84c982ac2564e6bcae1f984173194666a`;
  - fold 3: `6f26bc3049459f88ef8d1f06147baa587a2f739ecb8ed1985874c7368fcc8d0b`;
  - fold 4: `22968dc5dda176a69244cd1a28781bca17c86a00cc043895b7d1cee700515d79`;
- schema: only `schema_version`, `sample_key`, `candidate_success`;
- every key belongs to its public frozen fold;
- folds are pairwise disjoint and their union equals all 4,000 public keys;
- aggregate success statistics computed: false;
- selector metric computed: false;
- oracle metric computed: false;
- training started: false.

The five fold JSONLs remain ignored by Git and are covered by external per-file retention. `data/PRIVATE_LABEL_MANIFEST.json` is the committed authority.