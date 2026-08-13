# Amendment 014: Physical TriVUS Outer Runner

Date: 2026-08-12

Timing: after formal-primitives commit `2aa3c264a16239b419361be3857f200cb7f4264e`, before any formal TriVUS optimizer step or outer-label access.

## 1. Per-outer workflow

The launcher consumes one committed authorization nonce and creates a canonical receipt. Each outer worker revalidates that receipt against the committed authorization and exact implementation blobs, validates all locked dependencies and protected PID identity, then:

1. loads public/blind inputs and validates the complete context bank;
2. for each of four OOF holdouts, separately opens only two model-training folds, one checkpoint fold, and one OOF fold;
3. fits seven model specs with a shared split seed and predicts each spec's OOF scope;
4. persists seven canonical OOF prediction JSONLs and reloads each against an independently derived public/spec context-key scope;
5. composes TARGET_ONLY by exact family coverage and selects independent safe thresholds for JOINT3, TARGET_ONLY, JOINT2_NO_ANDROID, NO_VISUAL, and RANDOM_ID_PLACEBO from the reloaded artifacts;
6. fits seven final models on all four outer-development folds using per-spec half-up median epochs;
7. writes each model state and standardizer to disk and hashes them;
8. atomically fsyncs the pretest seal;
9. reassembles development data, reloads OOF artifacts, recomputes thresholds, and reloads every final artifact from disk;
10. only then opens the outer fold's VUS and Android labels once, verifies observed hashes against the seal, predicts, applies frozen thresholds, and writes success-bit outputs.

## 2. Pretest seal

The pretest includes exact outer/dev fold identities, observed hashes of all opened development labels, independently recomputed sealed outer-label hashes, all code/config/public/blind/context hashes, five threshold reports, complete checkpoint histories, deterministic hashes of every assembled training/checkpoint/OOF/final data object, seven persisted OOF paths/hashes/content hashes, four selected epochs and final epoch per spec, and seven relocatable final artifact paths/hashes. Outer-label loading rejects any schema, hash, scope, recomputation, or artifact mismatch.

## 3. Outputs

Outer result outputs five policies: JOINT3, TARGET_ONLY, JOINT2_NO_ANDROID, NO_VISUAL, and RANDOM_ID_PLACEBO. JOINT2 contains only Mind2Web and ScreenSpot-Pro. Other policies contain all three families. Values are held-out safe/direct/fallback success bits keyed by sample key. No action-semantic compatibility is claimed.

## 4. Concurrency and execution

Outer folds are independent and assigned one physical GPU each, exposed locally as `cuda:0`. Each worker writes only under a hidden nonce-scoped attempt directory. The launcher validates all five result/pretest/completion-marker hashes, then publishes the entire attempt with one same-parent directory rename to `formal/`. Failed attempts remain isolated and cannot be silently replayed under the consumed nonce. PID 2274 identity is checked before training, before pretest publication, before outer labels, and before final publication. The launcher never signals or reprioritizes external processes.

## 5. Execution boundary

Implementation, synthetic tests, launcher, and final adjudicator must be committed before a separate formal authorization. No worker may start before authorization binds every exact implementation blob.

Final bootstrap follows Amendment 008 literally: seed is `20260900 + control_offset + cell_index`; equal-cell/family/three-family arrays are composed at the same replicate-array index within each comparison. No post-result resampling change is allowed.