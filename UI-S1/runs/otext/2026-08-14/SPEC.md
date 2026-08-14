# OTEXT — 文本目标 OCR 通道 post-selection validation spec

Round: `otext`
Run dir: `runs/otext/2026-08-14/`
Date: 2026-08-14
Status: FROZEN BEFORE ANY OTEXT OCR OR LABEL STATISTIC
GPU: 零；OCR仅CPU

## 0. 证据地位与范围

ORTH已在同一1,581行ScreenSpot-Pro上查看两engine、宽matcher grid、text/icon accuracy、overlap与$\kappa$，并据此建议text-target OCR。因此同一数据上的OTEXT无法恢复confirmatory地位，即使重新生成OCR并nested评估也不能消除方向选择污染。

本轮严格降级为`POST_SELECTION_VALIDATION`：可产生预注册方法候选与附录证据，但不可单独支持论文方法主张。任何claim-eligible确认必须在未被ORTH使用的新数据上重新生成。本轮不改变F1、Q1、`TRIVUS_NOT_PROMOTED`、VUS-SR、SPLIT、MASK、CEIL、ORTH状态。

范围仅ScreenSpot-Pro全1,581行；不得用`ui_type`门控。text/icon只作held-out evaluation-side分解。

## 1. OCR与多重性

必须重新生成OCR raw，不复用ORTH raw。

- 主engine：EasyOCR 1.7.2，决定O-G1与O-P1至O-P5。
- replication：RapidOCR-ONNXRuntime 1.4.4，只报告同方向/异质性，不能救主engine。

engine/model/config hashes必须在OCR forward前commit preflight。两engine参数逐位复用ORTH preflight。OCR raw逐行write/flush/fsync，包含polygon/text/confidence，且不含instruction、GT或labels。

## 2. Nested split

沿用application GroupKFold(5)。每个outer fold $f$：

- outer test：fold $f$，选择期间禁止访问labels；
- inner validation：fold $(f+1)\bmod5$；
- inner train：其余三折；
- outer development：除$f$外四折，仅用于在选择完成后重估source reliability/priority，不重新选择门限。

Stage 0是五个outer fold各自的inner-train/inner-validation选择过程，不是一次全数据dev扫描。O-G1用五个inner-validation heldout predictions拼接后的全1,581-row-equivalent加权净收益：各fold inner-validation贡献按对应outer-test行数标准化加权。通过后才运行outer-test Stage 1。

## 3. GT-free literal extraction

抽取family顺序冻结为：

1. `quoted`：ASCII单双引号及Unicode弯引号内的非空substring；多段均保留；
2. `caps_camel`：空白/标点tokenize后，连续满足“全大写且至少2个字母”或camelCase/PascalCase含大小写转换的tokens；连续段拼空格；
3. `full_normalized`：完整instruction做Unicode NFKC、casefold、连续空白折叠、去首尾标点。

前两类无抽取结果则该family在该row不触发，不回退到全文。literal归一化后少于3字符则丢弃。family由inner validation选择。

## 4. 匹配与门

matcher family顺序：`exact`、`normalized`、`edit`。

- exact lexical similarity：OCR raw text trim后逐字包含于任一raw literal则1，否则0；
- normalized similarity：NFKC/casefold/空白折叠/首尾标点去除后substring则1，否则0；
- edit similarity：ORTH的token-contiguous span最大normalized edit similarity。

统一box score为`lexical_similarity × OCR_confidence`，范围`[0,1]`。每row取最高score box；tie依次取OCR confidence、normalized文本长度、top、left、engine raw order。box中心为OCR proposal。

对每个outer fold×engine×extractor×matcher，在inner-train非零row-best scores上构造12点$\theta$ grid：quantiles `[0.0,0.1,...,1.0]` 加`+infinity`。重复值保留原index但选择tie取更高index。无非零score则grid仅含11个0和`+infinity`并标记。

row进入OCR gate当且仅当extractor产生literal、存在OCR box且row-best score严格`>= theta`。不得使用GT、row class、`ui_type`或pool disagreement作为runtime gate。

## 5. Baselines与inner selection

原pool为C-uni 12 candidates，冻结slot order为views 0–3 × `[GTA1,Qwen3,UI-TARS]`。

两个强制主对照：

- `majority`：outer-development source-reliability priority，严格复用F1 ScreenSpot majority；
- `dev_selection`：在inner validation从E1七个methods `[majority,A0,ours,A1,A2,A3,A4]`按accuracy选择，tie取列表更早者；在outer development重估priority后应用已选method，严格复用CEV V4。

另报`best_single`：outer-development source reliability最高slot，不作判据。对ScreenSpot，majority与best-single实现可能相同，仍分别命名并记录。

对每个engine，extractor/matcher/$\theta$组合在inner validation计算Arm O相对majority与dev-selection的全row accuracy gain。选择objective为两项gain的最小值；最大化该minimum，tie依次取更高$\theta$ index、extractor顺序更早、matcher顺序更早。

O-G1仅由EasyOCR决定：五折selected inner-validation outputs按outer-test fold size加权后，`min(gain_vs_majority,gain_vs_dev_selection) >= 0.007`才进入Stage 1。否则O-K1，Stage 1不运行。选中$\theta$为quantile index 0或`+infinity` index 11即O-K7；不扩grid。

## 6. Arm O与controls

Arm O在gate rows用OCR proposal替换各baseline答案；gate外保持baseline。majority-origin和dev-selection-origin分别产生输出。

`blind_match`：同一selected extractor/matcher，$\theta$固定为grid index0，分别作用于两baseline。

`random_gate`：每outer test fold选择与EasyOCR主gate相同数量rows。row排序SHA-256(`row_id|outer_fold|20260814|random_gate`)；取最小者。每选中row从原12 candidates中按SHA-256(`row_id|outer_fold|20260814|random_candidate`)模12取candidate。一个冻结random control，不重复择优。

## 7. Arm F

`F_density`使用weighted complete-link B3：pairwise等价仍为official axis-aligned 14px complete-link与原candidate order。每个视觉candidate权重固定为$1.5936767669403409/12$；gate开启时OCR proposal作为第13候选，权重$w$。winning group按成员权重和、最早group order；代表候选按candidate weight、原order。OCR排在原12候选后。$w$ grid：`[0,0.125,0.25,0.5,1,1.5936767669403409,2,4]`。

每fold先按§5选择gate，再在同一inner validation选择$w$，最大化F_density相对majority与dev-selection gain的minimum；tie取更小$w$。outer test只应用selected $w$。

`F_majority`是13-source availability-aware priority control：在outer development上估计原12 sources及OCR source的correctness，OCR未gate rows视为source unavailable而非错误；test row从可用sources按reliability、原12顺序后OCR选择。它不使用$w$，与F_density分别报告。

## 8. Stage 0报告

每fold只报告inner train/validation：

1. $P(\text{OCR correct}\mid\text{baseline correct})$与$P(\text{OCR correct}\mid\text{baseline wrong})$，majority/dev-selection分别报；
2. selected setting的score deciles内`baseline correct/wrong × OCR correct/wrong`表；
3. 全冻结grid净收益curve，net按`min(vs majority,vs dev-selection)`和两项单独值报告；
4. selected extractor/matcher/$\theta$/w、所有validation scores、boundary flags。

这些量依赖inner-validation labels，只用于selection/evaluation，不进入runtime。

## 9. Stage 1与统计

O-G1通过后，五fold各自selected参数应用outer test，每row只产生一个cross-fitted prediction。10,000次99% percentile CI，fold内按application group有放回paired bootstrap。seed base `20260814`，endpoint/control offsets写入实现。

O-P1（主）：EasyOCR Arm O必须同时满足：相对majority的99% CI下界`>0`，且相对dev-selection的99% CI下界`>0`。全1,581 rows估计，不得只报gate。

O-P2：F_density和F_majority相对两个强制baseline分别报告；只有四项CI下界均`>0`才称融合通过。

O-P3：held-out gate内$P(\text{OCR correct}\mid\text{baseline correct/wrong})$及差值，描述性。

O-P4：gate内Arm O与baseline accuracy，描述性。

O-P5：按`ui_type=text/icon`分解，evaluation-side。

RapidOCR完整复现同一nested流程与终点，但不改变EasyOCR主判定。

## 10. Kill conditions

- O-K1：EasyOCR O-G1 `<0.70pp`，Stage 1前终止；
- O-K2：EasyOCR O-P1任一主对照CI含零或为负；
- O-K3：EasyOCR Arm O相对冻结random_gate的paired CI含零；
- O-K4：EasyOCR Arm O相对blind_match的paired CI含零；主张改为OCR presence而非match quality；
- O-K5：F_density通过而F_majority未通过，记absorbed，不只报density；
- O-K6：EasyOCR与RapidOCR O-P1定性不同，降级engine-specific；
- O-K7：任一主fold selected theta在grid index0或11，记grid failure，不扩grid。

失败后不得换engine、extractor、matcher、score、weighting、baseline或control再报。

## 11. 留存与执行顺序

五个已泄漏SSPro cells不得作为优化目标。本轮重新生成的两engine raw OCR、SSPro dataset snapshot manifest、外部inputs、derived rows全部纳入SHA-256与独立备份 `/scratch/workspaceblobstore/otext/2026-08-14`。不复制已由上游manifest锁定的3.36GB图片，但逐图hash重验且backup manifest引用上游dataset snapshot；禁止递归删除raw OCR/derived rows。

顺序：

1. commit SPEC与`configs/otext_prereg.yaml`；
2. input/engine/model/dataset preflight commit；
3. raw OCR writer implementation commit；
4. 重新执行两engine OCR并锁raw manifests；
5. Stage 0 implementation commit；
6. 执行Stage 0，按EasyOCR O-G1；
7. 若通过，commit selected-parameter manifest后执行Stage 1；
8. 按O-P3、O-P1、O-P2、O-P4、O-P5和controls报告；
9. retention、STATUS、push。

无论结果如何，报告必须以`POST_SELECTION_VALIDATION`标注；真正confirmatory需要新untouched数据。