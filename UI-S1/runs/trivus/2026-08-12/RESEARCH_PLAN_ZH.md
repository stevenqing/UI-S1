# TriVUS：三 benchmark 可变 action-set utility selector

## 为什么不是继续堆 fusion

VUS-SR 已证明 fallback-relative set utility learning 有效；CARE、RAVEL、DELTA、CIVA 则共同说明，固定容量 routing、pixel fusion、channel fusion 和 raw-direct admission 都不能自动提高最终安全策略。

下一步保留 VUS-SR 的正确部分，但移除“永远 12 candidates”的数据假设。Mind2Web/ScreenSpot-Pro 使用现有 $K=12$ banks，AndroidControl 使用真实三模型 $K=3$ action pool。共享模型只看 action、geometry、disagreement 和 blind visual evidence，不看 source/model/slot identity。

## 先补数据

AndroidControl 当前三模型交集只有 Low 1,096 / High 1,056 rows，且单模型相对完整历史集有 2--3 pp bias。不能直接在这个子集上优化。

R0 使用原模型、原 prompt/parser、原 processor、原 row order 和 `--resume` 补齐：

- GUI-R1 Low/High：缺 904/944；
- UI-R1-E Low/High：缺 176/208；
- UI-AGILE 已经 2,000/2,000。

恢复阶段禁止计算 accuracy 或挑方法。四条 lane 全部 2,000 rows、identity/provenance/hash 完整后，才计算一次三候选 oracle headroom。

## 主方法

TriVUS 是 masked permutation-equivariant set ranker，支持 $K=3/12$。默认保留冻结 fallback，只有 candidate utility 和 downside 同时通过 nested threshold 才覆盖。

JOINT3 必须同时满足：M2W/SSPro 不劣于 VUS-SR、AC Low/High 不劣于 majority、至少一族显著提升、三族 balanced CI 为正，并超过 TARGET_ONLY 和 NO_VISUAL。否则停止，不用某一数据集的增益掩盖另一数据集退化。