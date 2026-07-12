# Pass@8 Candidate Diversity With a Stronger Multimodal Corrector

## Research question

The earlier experiments established two distinct Pass@8 signals:

1. **Full-task Android Control sampling**: UI-TARS-7B improved from 13.48% Pass@1 to 20.61% Pass@8, while Qwen2.5-VL-7B improved from 6.67% to 16.05%. The UI-S1 result was only available on 385/1,543 tasks and is not directly comparable.
2. **GUI-360 critical-step complementarity**: over 962 previously selected critical steps, the four-source candidate union had 30.77% high-quality coverage versus 4.89% for a matched 32-sample single-SFT reference, a +25.88pp diagnostic margin. Union any-correct coverage (65.28%) did not exceed the single-SFT any-correct reference (67.05%).

These results show that useful actions often exist in the sampled candidate set. They do **not** show that a deployable model can identify those actions without ground truth. The new hypothesis is therefore:

> A substantially stronger multimodal model can convert Pass@8 candidate diversity into positive student-relative rescue-minus-regression utility by selecting among a fixed, anonymized candidate packet.

## Paired model comparison

- Current selector: Qwen3.5-9B, native multimodal checkpoint.
- Strong selector: Qwen3.5-35B-A3B, native multimodal MoE checkpoint.
- Both receive the identical screenshot, task goal, teacher-forced prior-action history, and fixed candidate packet.
- Both use the same deterministic fixed-choice prompt and decoding configuration.
- The selector may choose `BASELINE`, but may not invent or alter an action.
- Invalid output and request failure fall back to `BASELINE`.

Two label-blind zero-GPU controls are also generated before dev or locked labels are unsealed: exact-action plurality (requiring at least two exact votes) and cross-source consensus (requiring anonymized agreement from at least two generators). These diagnose whether model scale contributes beyond support counting; they are not part of the predeclared stronger-model claim.

The candidate packet combines the first eight cached samples from each of four sources: SFT anchor, Qwen3-VL-8B, Qwen3.5-9B, and LLaVA-1.5-7B. Exact actions are retained so coordinate quantization cannot remove an oracle candidate. The packet exposes exact support, nearby-action support, and anonymized independent-source support, but not source identities.

## Frozen split and leakage contract

The selector holdout was frozen before either selector was run:

| split | episodes | steps | purpose |
|---|---:|---:|---|
| smoke | 12 | 23 | parser/runtime validation only |
| dev | 133 | 231 | paired development diagnosis |
| locked test | 398 | 708 | one-shot confirmatory gate |
| total | 543 | 962 | fixed critical-step population |

Splitting is deterministic and episode-disjoint. Blind packets exclude:

- ground-truth actions;
- candidate correctness and matcher rewards;
- generator/model identity;
- GT-derived criticality and confidence diagnostics.

Labels and candidate provenance are stored separately. The locked labels may be read only after both current and strong outputs are complete and packet hashes match.

Both dev prediction files were completed without opening dev labels. Locked predictions were then launched immediately, so the confirmatory outputs cannot incorporate dev-label feedback.

This split is **selector-fresh, not benchmark-fresh**. The underlying GUI-360 test episodes were evaluated in earlier studies, and the 962 targets were originally selected using GT-derived critical-step diagnostics. Conclusions must therefore be scoped to fixed-choice selection on this pre-existing hard-step population.

The study does not solve online all-step routing: this hard-step population contains very few frozen-student-correct controls. A positive result would validate offline candidate recovery on targeted hard states, not prove safe replacement on arbitrary states. Any subsequent training must retain clean replay and must still pass a full-policy held-out causal evaluation.

## Predeclared metrics and gate

For frozen-student correctness $M(a)\in\{0,1\}$, per-step utility is

$$u_t=M(a_t^{selector})-M(a_t^{student}).$$

The primary population metric is

$$U=\frac{N_{rescue}-N_{regress}}{N_{steps}}.$$

The proposal-to-selection conversion metric is oracle-headroom capture:

$$C=\frac{Acc_{selector}-Acc_{student}}{Acc_{packet\ oracle}-Acc_{student}}.$$

Both the frozen student action and the selected action use the same existing GUI-360 reward matcher with correctness threshold 0.85.

The confirmatory selector gate passes only if all conditions hold on locked test:

1. $U>0$;
2. rescue count exceeds regress count;
3. the episode-cluster bootstrap 95% lower bound for $U$ is above zero.

The stronger-model claim additionally requires the episode-cluster bootstrap 95% lower bound of strong-minus-current selected accuracy to be above zero. No policy training is authorized unless the selector gate passes. If it passes, the only pre-authorized training recipe is the previously validated 25% selected revision plus 75% clean replay arm.

## Compute and checkpoint validation

All new GPU work is restricted to physical GPUs 4–7. Physical GPUs 0–3 are forbidden, and protected PID 1911 must remain alive.

- Qwen3.5-9B runs on physical GPU 4 with tensor parallel size 1.
- Qwen3.5-35B-A3B runs on physical GPUs 4,5,6,7 with tensor parallel size 4.
- Upstream checkpoint revisions are Qwen3.5-9B `c202236235762e1c871ad0ccb60c8ee5ba337b9a` and Qwen3.5-35B-A3B `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- vLLM uses `gpu_memory_utilization=0.65`, an explicit 8 GiB KV cache per worker, eager execution, and one image per prompt.
- The 35B checkpoint contains 14 indexed BF16 safetensor shards totaling 71,903,878,016 bytes. No indexed shard is missing.
- The strong server successfully loaded with 16.52 GiB model memory per GPU under vLLM 0.23.0.

## Smoke result

Both selectors produced 23/23 structurally valid outputs with 100% parse rate and no fallback. The current selector changed 65.22% of steps and the strong selector changed 52.17%. On this very small hard-step smoke split:

- frozen student accuracy: 0%;
- packet oracle accuracy: 34.78%;
- current selected accuracy: 0%;
- strong selected accuracy: 0%.

Thus the smoke test validates the pipeline and confirms nontrivial oracle headroom, but provides no evidence that either model can capture it. The prompt is frozen after smoke; no smoke-label-driven prompt tuning is allowed before dev or locked evaluation.

## Final results

Dev predictions and locked predictions were both completed before either split's labels were opened. The result is stable from dev to locked test:

| selector | dev utility | locked utility | locked 95% episode-cluster CI | locked rescue/regress | oracle capture |
|---|---:|---:|---:|---:|---:|
| Qwen3.5-9B current | +6.49pp | **+6.36pp** | **[+4.53pp, +8.31pp]** | 46 / 1 | **20.93%** |
| Qwen3.5-35B-A3B strong | +2.60pp | +4.52pp | [+2.93pp, +6.20pp] | 33 / 1 | 14.88% |
| exact plurality | +3.03pp | +3.25pp | [+1.87pp, +4.73pp] | 25 / 2 | 10.70% |
| cross-source consensus | +5.63pp | +5.37pp | [+3.61pp, +7.25pp] | 39 / 1 | 17.67% |

On locked test, frozen-student accuracy is 0.42% and packet-oracle accuracy is 30.79%. Qwen3.5-9B raises selected accuracy to 6.78%, while Qwen3.5-35B-A3B reaches 4.94%.

The primary proposal-to-selection hypothesis is supported: non-oracle fixed-choice selection converts a statistically positive fraction of Pass@8 headroom into student-relative rescue. The model-scaling hypothesis is not supported. Strong-minus-current accuracy is -1.84pp with 95% CI [-3.69pp, +0.00pp], with 13 strong-only correct steps versus 26 current-only correct steps. The 35B model is also more conservative, changing 34.18% of actions versus 60.03% for 9B.

Cross-source consensus reaches +5.37pp and is not distinguishable from the 9B selector in their direct paired comparison (-0.99pp, 95% CI [-2.98pp, +1.12pp]). Therefore, much of the usable signal comes from repeated independent proposal support; increasing corrector scale does not explain the gain. Qwen3.5-9B still recovers more total oracle headroom, but its advantage over consensus is not locked-significant.

The predeclared selector gate passes. This authorizes preparation of a **new train-split** arm using 25% selected revisions and 75% clean replay. No dev or locked-test row from this study may enter training, and full-policy held-out evaluation remains mandatory.

## Interpretation boundary

A positive packet oracle is only an existence result. The scientific question is whether visual and historical evidence is sufficient for a non-oracle selector to recover that headroom without regressing correct student actions. A negative result would sharpen the earlier conclusion: candidate diversity exists, but current VLM reasoning—even at substantially larger scale—does not reliably identify useful revisions.
