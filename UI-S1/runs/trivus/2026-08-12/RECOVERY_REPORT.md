# TriVUS R0 AndroidControl Recovery Report

Date: 2026-08-12

Outcome: `PASS_TRIVUS_R0_RESULT_BLIND_RECOVERY`

## Coverage

All six frozen AndroidControl lanes now contain exactly 2,000 unique rows in exact reference order:

| Lane | Rows | Recovery status | SHA-256 |
| --- | ---: | --- | --- |
| UI-AGILE-7B Low | 2,000 | preexisting two-shard complete | `de7518dc...` / `63aab88e...` |
| UI-AGILE-7B High | 2,000 | preexisting two-shard complete | `a4f2298f...` / `b239a834...` |
| GUI-R1-7B Low | 2,000 | resumed from 1,096 | `62d0d07a395753f6a021ada36d7b1a91d9646113b41be947cfc462692f9600ab` |
| GUI-R1-7B High | 2,000 | resumed from 1,056 | `e8a5d40fbd1c15b192555eddf85090f77a634f71f427f54662fade793ebc628f` |
| UI-R1-E-3B Low | 2,000 | resumed from 1,824 | `80b43fc4c633ca71e000ddc95826452202409cb75b26d7cd488c51323cf560a8` |
| UI-R1-E-3B High | 2,000 | resumed from 1,792 | `246c9bdde73bc9b35f7668d699d79d997303849b8d402ffc0a2f6cc6e90ce26c` |

All lanes have ordered row-identity SHA-256 `0669928f4dfa0852472a39819f45f0ee88c7cc889969aeb833e749ad2b08d2cd`. Low and High use the same 2,000 paired row IDs, 1,039 episode groups, image identities, and frozen folds.

## Integrity

- source script, roster, sample manifests, model revisions, model-index hashes, historical seed bytes, prompt hashes, prediction schema, shard identity, source/image provenance, and stable-index coverage passed;
- four child logs ended with `status=PASS` and no traceback/error/exception;
- historical partial files remain byte-identical to their preregistered hashes;
- recovery outputs were written only under `runs/trivus/2026-08-12/recovery/` with per-row fsync;
- protected PID 2274 retained the frozen start ticks, comm, command-line hash, and executable before and after recovery.

Reference rows contain historical `gt_*` fields, but R0 code never indexed those keys. No scorer/evaluator was imported and no candidate success, accuracy, oracle, or aggregation was computed.

R0 authorizes only the already-frozen R1 complete-bank oracle-headroom gate. It does not authorize blind selector inference or TriVUS training.