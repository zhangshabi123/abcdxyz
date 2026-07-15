# 精度匹配机制设计：分组比较器体系（v1）

> 适用范围：agentic_convert_bench 的 31 个模型（19 个高转换难度模型 + 12 个分类骨干对照集），
> Torch→ONNX 赛道起步，设计目标是零改动泛化到 TFLite / CoreML / TensorRT 与 fp16 / INT8 精度档。
>
> 本文档由三份独立设计（数值保真优先 / 反作弊优先 / 工程泛化优先）+ 一轮对抗批判综合而成，
> 所有分歧点的裁决理由见各节标注。日期：2026-07-15。

---

## 0. 背景与问题

当前 harness 是 KernelBench 时代的设计：参考 = 原始 PyTorch 模型，候选 = 转换产物，同权重、
秘密随机输入，判据 `|a−b| ≤ atol + rtol·|ref|`，普通输出全元素必须通过，大输出走
nRMSE + mismatch 率。已知问题：

1. **全元素门对大张量太脆**：99.6% 元素通过仍 FAIL，换 seed 结果 ±1/15 波动；
2. **真实模型的输出不是裸张量**：检测模型输出变长框集合（shape 都对不齐）、关键点来自热图
   argmax（整像素跳变）、MoE 路由和 LSH 哈希是离散决策（微扰翻转 → 局部大偏差但转换忠实）；
3. **朴素比较对语义作弊不设防**：YOLO-World 把文本嵌入烘焙成常量后，同词表下输出逐位一致，
   纯精度比较 100% PASS，而开放词汇能力已静默丢失。

核心问题：**每个模型输出都不一样，要不要分组匹配？怎么分才能 generalize？**

## 1. 结论：要分组，按「输出语义 × 驱动协议」分

分组的对象**不是模型，也不是任务类别**。正确的原子单位是模型的**输出槽**（output slot）：
每个输出槽绑定一个（比较器, 驱动协议）组合，"组"是绑定组合的推导等价类。一个模型可以
横跨多个组（Mask R-CNN 的 pre-NMS 特征走稠密门、框集合走集合匹配、掩码走实例掩码门）——
这正是双门反作弊的来源。

### 1.1 为什么任务类别（det/seg/nlp）是错误维度——bench 内部反例

- **同任务 ≠ 同比较器**：同为"检测"，rtdetr / owlvit / groundingdino 输出固定 query 数的
  稠密 logits+boxes（直接按 index 统计比较），maskrcnn / keypointrcnn 输出变长集合（必须
  置换不变匹配）——同一标签下是两套完全不同的门。
- **异任务 = 同比较器**：mask2former（实例分割）的 raw 输出与 DETR 族检测头在比较学上同类
  （固定 100 query 槽位，槽序确定）；birefnet（分割图）与 longformer 的 hidden states 用
  同一个大稠密张量统计门。
- **NLP 组内部是协议异构而非度量异构**：flan_t5 要 teacher-forced 多步、rwkv 要带状态分块
  驱动、deberta 是单前向——比的都是稠密 logits，差别全在驱动方式。

### 1.2 两个极端为何被否决

- **完全不分组（一把尺子）**：双向同时失效。误杀侧——对 maskrcnn 连 shape 都对不齐，对
  upernet 3.9 千万元素的全元素门统计上必挂（99.6% 仍 FAIL 的实证）；漏放侧——对 yoloworld
  烘焙词表这种逐位一致的语义丢失 100% 误放。
- **每模型手写脚本**：31 份互异判据无法做统一元测试（ref-vs-ref 必 PASS、已知坏产物必
  FAIL）、无法声称跨模型可比的 PASS 语义、每份脚本都是会读 harness 源码的 agent 的独立攻击
  面，×4 格式 ×3 精度后维护量爆炸。
- **诚实声明**：分组并没有消灭全部 per-model 工作——reformer 的 LSH 种子 wrapper、bigbird
  的配置断言、sam2 的模式钉死、t5 的多图接线、rwkv 的分块驱动，约 5-6 个模型仍需 harness
  侧代码。分组的真实价值是**共享比较器语义、共享校准地板、统一反作弊层**，不是零边际成本。

## 2. 总体架构：三层，只有中间一层是"分组"的

```
第 0 层  harness 宪法 C1–C8（全模型、全格式、不可协商、不分组）
第 1 层  比较器库（3 比较器 × payload 插件 + 4 驱动协议；阈值按 (format, precision) tier 取值）
第 2 层  每模型 manifest（harness 侧冻结的 YAML：导出契约 + 槽位绑定 + 电池轴）
```

执行管线（换格式只动第一层适配器）：

```
Artifact ──RunnerAdapter(format)──> run(inputs)->outputs     # 唯一知道 ONNX/TFLite/CoreML/TRT 的层
(torch_ref, artifact_fn) ──ProtocolDriver──> [(ref_bundle, cand_bundle), ...]
bundle ──Canonicalizer(manifest)──> fp64 数组、规范 layout、规范槽名、INT8 已反量化
Comparator.compare(ref, cand, spec, tier) ──> GateResult      # 纯 numpy，禁 import 格式库
Aggregator(全部门 × 全部 trial) ──> ModelVerdict
```

### 2.1 第 0 层：harness 宪法（最高优先级）

历史上两次得手的作弊（fp16 假转、文本嵌入烘焙）都不是比较器数学问题，而是这一层缺失。
比较器再精巧，宪法有洞就全盘皆输。

| # | 检查 | 针对的作弊 |
|---|---|---|
| C1 | 产物真实性普查：initializer dtype 直方图（fp16 赛道 ≥99% 权重字节 fp16）、文件体积 ≈ 2B/param、INT8 赛道量化算子覆盖率——解析文件本体，不信元数据 | fp16 假转、INT8 假量化 |
| C2 | "好得不真实"下界：声称 fp16/INT8 的产物与 torch fp32 的差异低于真实降精度舍入下界 → 真实性 FAIL（双侧带：既要 > 舍入下界又要 < 容差上界） | 假降精度的数值指纹 |
| C3 | 导出契约强制：输入名/dtype/动态轴逐一实跑比对，契约外输入或缺输出名 → 直接 FAIL，不进数值比较 | 只导简单子图、外挂输入通道、烘焙掉 text 输入 |
| C4 | 沙箱执行：stock runtime、无网络、评测进程禁 import torch、拒绝产物内嵌自定义 Python op | 把 torch 藏进 runner |
| C5 | 秘密电池 + 丰富度断言：检测/分割电池必须是留出的**自然图像**（高斯噪声图上检测器输出空集 → 集合匹配空对空平凡通过），设计期断言每图参考输出 ≥K 框、Switch 电池 8 专家各 ≥50 token、开放词汇电池含 held-out 类名；断言做成随 bench 版本跑的 CI | 对可见样例过拟合、空对空平凡 PASS |
| C6 | 参考确定性审计：torch 参考双后端/双进程跑出噪声地板，所有门槛必须 > 地板；reformer LSH 种子、bigbird numpy 随机块在 harness wrapper 内钉死 | （防误伤 + 防 agent 拿"参考不稳"当申诉借口） |
| C7 | NaN/Inf/shape 政策 + 反常量：转换输出 NaN/Inf 或 shape 错 → FAIL；稠密输出跨电池方差与参考方差之比 ∈ [0.5, 2]（仅稠密输出生效，集合值输出未定义） | 常量/近常量输出骗聚合指标 |
| C8 | 条件敏感性差分：对每个条件轴（文本 prompt / 点提示 / global mask / 专家激活），取同一底输入 x 与两个条件 c1≠c2，门为 cosine(Δconv, Δref) ≥ 阈 且 ‖Δconv‖/‖Δref‖ ∈ 带内，**前置条件 ‖Δref‖ ≥ 下限**（设计期挑选高分离度条件对，防除小数爆炸） | 一切"烘焙成常量"类作弊的结构性克星；主门放宽的 fp16/INT8 档尤其关键 |

### 2.2 第 1 层：比较器库

**3 个核心比较器**（取代早期草案的 6 个单体——单体化是 keypoint 这类"第 N 个特例"的来源）：

| 比较器 | 覆盖 | 说明 |
|---|---|---|
| `dense` | 小张量全元素、大张量统计、逐步 logits 的度量部分 | 单一连续判据（§3.1），无 10⁴ 元素分流开关 |
| `matched_set` | 变长检测集合、实例掩码、关键点 | 匹配器（hungarian_iou / hungarian_l2）× payload 插件（box / label_exact / scalar / dense_per_instance / keypoints），§3.2 |
| `discrete_agreement` | argmax / top-k / 路由决策 | 一致率 + ref 侧 margin 豁免（§3.3）；不是独立存在——必须附着在产生该决策的分数场上 |

**4 个驱动协议**（从比较器中拆出——协议决定"怎么产生可比张量束"，不做数值判断）：
`single_forward`、`teacher_forced_decode`（flan_t5 / switch）、`state_carry_scan`（rwkv，
变 chunk 长带状态交接）、`prompted_forward`（sam2 及可提示模型，变点数/有无 box 提示）。

### 2.3 第 2 层：manifest（§5 给字段草案）

harness 维护者编写、随 bench 版本 hash 冻结、agent 只读。声明三件事：导出契约（输入签名、
动态轴、必须暴露的输出槽含 aux 槽）、槽位 →（比较器, 协议）绑定与结构参数、电池轴要求。
**数值阈值一律不进 manifest**——否则 31 份 YAML 各调各的阈值，跨模型可比性与反作弊审计全毁。

## 3. 比较器规范

### 3.1 `dense`：单一连续判据（消灭 10⁴ 元素悬崖）

对任意元素数 N 的张量，PASS 当且仅当同时满足：

- **(a) 失配预算**：`#{i : |Δᵢ| > atol + rtol·|refᵢ|} ≤ ceil(q·N)`，q = 0.1% 起步。
  ceil 使小张量预算自动收缩到 1，无模式开关，9,999 元素错 1 个全挂 / 10,001 元素错 100 个
  能过的病态不复存在；
- **(b) 严重度硬顶**：∀i, `|Δᵢ| ≤ C·(atol + rtol·|refᵢ|)`，C = 10 起步。防"预算被少数粗大
  误差滥用"——Longformer 式 70%→17% 的真崩坏表现为少数元素的大误差，(a) 单独会漏，(b) 抓住；
- **(c) 聚合门**：`nRMSE = RMS(Δ)/max(RMS(ref), atol/rtol) ≤ T`。分母下限使 ref≈0 时退化为
  绝对误差口径，消除边界 logits 过零处的 rtol 病态（birefnet 边界、rtdetr 背景 query）；
- **(d) 分块反稀释**：按 16×16 像素块 / token 块计算块级 nRMSE，p99.9 块 ≤ 10·T。错误不能
  藏在全局平均里（birefnet 低分辨率冒充、局部炸掉的产物在此暴露）。

三者角色正交：(a) 容忍极值统计的必然冤枉（max-of-N ~ σ√(2lnN)），(b) 封死"错得少但错得
离谱"，(c) 封死"每元素都压线的系统性漂移"，(d) 封死空间聚集。

### 3.2 `matched_set`：匹配器 × payload

1. 剥离 padding（efficientdet 零填充行）；
2. 匈牙利匹配，代价 = 1−IoU，label 不等记 ∞（label 翻转可按类 logit margin 豁免）；
3. 配对项逐 payload 判：`box`（IoU ≥ 阈）、`label_exact`、`scalar`（score 走 dense 包络）、
   `dense_per_instance`（实例 soft mask logits 走 §3.1 + 二值 IoU 阈）、`keypoints`
   （OKS ≥ 阈或逐点 L2 ≤ 2px 起步，热图峰值 margin < δ 的关键点豁免——0.5px 这类数字在
   拿到 ref-vs-ref 地板前不允许写进配置）；
4. **未配对项政策**：只按 **ref 侧**分数是否落在后处理阈值 ± 带内豁免（candidate 侧分数无
   发言权，否则 agent 调自己的阈值把差异框挤出输出）；precision/recall 双侧记账（框洪水在
   precision 侧 FAIL）；契约钉最大输出数。
5. **级联漏网预算**（对抗批判的关键修正，见 §3.3）：R-CNN 族的 RPN→NMS→RoI 是决策级联，
   一个在 RPN 层合法豁免的近平分交换可在最终输出产生**不在任何豁免带内**的高分未配对框。
   "非豁免未配对 = 0"这种硬门会让 seed 波动复活。修法：未配对/翻转预算 = torch 双后端
   （CPU vs MPS/CUDA、eager vs compile）**全管线**实测的良性集合不稳定率 × 安全系数。

### 3.3 `discrete_agreement` 与 margin 豁免的适用边界

对决策场 s_ref 上的离散算子（argmax / topk / score 阈值 / NMS keep / top-1 路由），一次
不一致可豁免当且仅当 ref 侧 margin `m = s_ref(i) − s_ref(j) ≤ ε`，
`ε = 2·(atol + rtol·max(|s_ref(i)|, |s_ref(j)|))`（两分数可反向各漂一个包络；全部用参考
分数计算，agent 无法操纵）。豁免率 ≤ cap（地板校准 ×3）。

**硬门（非豁免翻转 = 0）只允许放在决策面可直接观测的地方**：分类 top-1、语义分割像素
argmax、导出了 router logits 的 Switch 路由。**不允许**假设豁免可以沿级联/注意力传播——
Switch 的路由翻转改变该 token 的 FFN 输出后，后续注意力把扰动混进所有其他 token；Reformer
的桶翻转同时改变新旧两桶内所有 token 的注意力集合。这些位置一律用"全管线地板 × 安全系数"
的校准预算，不用零容忍硬门。

## 4. 双门原则（Gate A / Gate B）

- **Gate A（决策前连续面）**：manifest 指定的最大固定形状、离散化前张量——DETR 族 decoder
  logits（天然固定形状，强制）、EfficientDet / YOLO-World pre-NMS 头（强制）、R-CNN 族到
  RPN/FPN 特征层（RoI 头依赖数据依赖 proposal，Gate A 只到此为止）。抓数值漂移藏进匹配
  模糊度的静默错误（RT-DETR 的 GridSample TFLite 发散正是此形态）。
- **Gate B（端到端语义面）**：后处理输出走 matched_set / discrete_agreement。抓后处理段
  转换错误；没有它，agent 只导 backbone 即可通过 Gate A（aux 槽缺名会先死在 C3）。
- 两门都过才 PASS。**论文主指标只看 Gate B / 端到端行为门**；Gate A、stateful 契约、多图束
  归类为反作弊门，由 harness 同一 wrapper 对 agent 与 baseline 同等施加，Gate A 失败单列
  不计入主成功率（baseline 转换器天然给不出 router logits 这类辅助输出，强行同契约会把
  "baseline 转不出"与"baseline 转错"混淆）。

## 5. Manifest schema 草案（字段级）

```yaml
schema_version: 1
model_id: hard_det_yoloworld_v2s

reference:                          # 参考模型如何构造——防"参考本身不确定"
  loader_id: ultralytics.yoloworld_v2s      # harness 侧注册的工厂 id
  weights_sha256: "..."
  pinned_config: {}                 # 语义关键配置钉死，如 bigbird: {attention_type: block_sparse}
  determinism:
    frozen_stochastic_params: []    # reformer: LSH 随机旋转；bigbird: numpy 随机块索引

export_contract:                    # 结构门（C3）：数值比较之前先过
  bundle:                           # 产物是"图束"，允许多图（t5: encoder/decoder/decoder_with_past）
    graphs:
      - name: text_encoder          # CLIP 文本塔必须在束内——契约输入是 token ids 不是嵌入
        inputs:  [{name: text_ids, dtype: int64, dynamic_axes: {n_class: [2, 32]}}]
        outputs: [{name: txt_feats}]
      - name: detector
        inputs:  [{name: image, dtype: fp32, layout: NCHW}, {name: txt_feats}]
        outputs:
          - {slot: raw_head, shape: [1, "4+K", 8400]}            # Gate A 载体
          - {slot: detections, role_hint: post_nms}              # Gate B 载体
    wiring: [text_encoder.txt_feats -> detector.txt_feats]       # 声明式连线，harness 执行，禁 agent runner 脚本
  dtype_policy: {fp16_tier: <见 §9 政策决定>}
  forbidden: [network_access, custom_python_ops]

protocol:
  driver: single_forward            # | teacher_forced_decode | state_carry_scan | prompted_forward
  driver_params: {}                 # rwkv: {chunk_lens: [1, 8, 64], total_lens: [17, 64, 193, 512, 1023]}
  battery:
    trials: 5
    input_generator: natural_coco_like     # harness 侧生成器 id；秘密 seed 由 harness 注入
    richness_asserts: [min_ref_boxes_per_image: 5]               # C5，设计期 CI
    axes:                           # 必须被激励的动态轴——写进契约就必须真的动态实跑
      - {axis: vocab, kind: conditioning, min_values: 4, holdout: true}
      - {axis: n_class, kind: shape, values: [2, 8, 17]}         # 至少 1 个值评测期才揭示

comparison:
  gates:
    - {id: gate_a, slots: [raw_head], comparator: dense, required: true}
    - id: gate_b
      slots: [detections]
      comparator: matched_set
      params:                       # 只有结构参数，无数值阈值
        matcher: {kind: hungarian, cost: iou, min_iou: 0.5}
        payloads: {boxes: box, labels: label_exact, scores: scalar}
        unmatched_policy: ref_score_threshold_band
      required: true
  probes:                           # C8 探针与门分离
    - {kind: sensitivity, vary: vocab, gate: delta_alignment}
verdict: all_required_gates_pass_all_trials
```

治理规则（防 manifest 蔓延成 YAML DSL）：结构参数只能从枚举库里选；数值阈值禁止进
manifest；每个 per-model 特例登记理由并计数——**特例超过模型数 20% 说明分类学错了**，
触发比较器库重审而不是继续加参数。每份 manifest 必须过元测试（ref-vs-ref 必 PASS、
合成坏产物必 FAIL）+ 独立评审。

## 6. Tier 阈值体系

每档一份版本化 YAML，比较器只读参数。tier 可对比较器做模式重映射但不能换比较器。
**下表全部数字是 placeholder，校准前无效**（§10 执行顺序）；fp32/fp16 起步值取自 ORT
官方套件先例，集合门阈值无任何工业先例可引，必须自校准。

| 参数 | fp32 | fp16 | int8 |
|---|---|---|---|
| dense 包络 (rtol, atol) | 1e-3, 1e-5 | 1e-2, 1e-3 | stat-only 模式 |
| 失配预算 q / 严重度 C | 0.1% / 10 | 0.1% / 10 | — |
| nRMSE T | 1e-3 | 1e-2 | 以 harness 侧参考量化步长为单位 |
| 决策一致率 / 豁免 cap | 99.9% / 地板×3 | 99% / 地板×3 | 决策门成为主门 |
| 配对框 IoU / 实例掩码二值 IoU | 0.99 / 0.99 | 0.95 / 0.97 | 0.90 / — |
| 关键点 | OKS 或 ≤2px（热图 margin 豁免） | 放宽一档 | — |

**INT8 专项修正**（对抗批判裁决）：阈值单位**不能**读产物自带的量化参数——scale/zero-point
是 agent 可控元数据，声明一个粗 scale 就能按比例放宽自己的门。以 harness 侧 fake-quant
参考或输出动态范围推导的步长上界为准；产物声明的 q 只做一致性校验（q 与观测输出格点间距
不符 → 真实性 FAIL，反而成为反作弊信号）。

**fp64 仲裁通道**：任何门以 <2× 阈值余量 FAIL 时，若 `err(conv vs fp64) ≤ 2·err(torch_fp32
vs fp64)` 则改判 PASS-by-arbitration，单列报告；只能 FAIL→PASS 不能反向，触发即落盘人工
审计（防仲裁通道被当第二判据钻）。fp64 不可行的模型（switch 620M、birefnet fp16 权重上转
语义存疑）退化为 torch CPU-vs-MPS 双路径地板，manifest 标注。

## 7. 31 模型分组表

| 组 | 门 | 模型 |
|---|---|---|
| **G1 dense 小张量** | 全元素包络（预算收缩到 1）+ top-1 标签零翻转（fp32 档） | 12 分类骨干：alexnet, vgg19, googlenet, squeezenet, shufflenet, densenet201, resnet101, regnet, mobilenet_v2, vit_b_16, swin_v2_t, cvt |
| **G2 dense 大张量** | §3.1 统计判据 + 分块 p99.9；upernet 加像素 argmax 一致率（margin 豁免）；battery 变长度/变分辨率 | birefnet, clipseg, upernet_swin_t, longformer, bigbird, deberta_v3_small, reformer† |
| **G3 固定 query 双门** | Gate A: raw logits/boxes 按 index dense（groundingdino 仅有效 token 列）；Gate B: 后处理 matched_set | rtdetr, owlvit, groundingdino, yoloworld_v2s, mask2former |
| **G4 变长集合双门** | Gate A: 契约强制 aux 槽（RPN/FPN、anchor 头）dense；Gate B: matched_set + 级联漏网校准预算 | maskrcnn_v2, keypointrcnn, efficientdet_d0 |
| **G5 序列/状态协议** | teacher-forced 逐步 logits dense + top-1 门 + 位置漂移斜率门（末 10% nRMSE ≤ 3× 首 10%，抓误差随位置累积） | flan_t5_small, rwkv4_169m, switch_base8 |
| **G6 带提示分割** | prompted_forward 协议；mask logits 按固定 token 槽 index dense + iou_predictions | sam2_tiny |
| **C8 敏感性差分（跨切）** | 叠加于上述各组 | owlvit, yoloworld, groundingdino, clipseg, sam2, longformer, switch, bigbird, reformer, rwkv, flan_t5 |

† reformer 是三份设计分歧最大且全体低估的模型：即便钉死 hash 种子，微小数值差仍会翻转
近平分投影的桶归属，且一个 token 换桶改变**两个桶**内所有 token 的注意力集合，任何 per-token
预算都可能被 2-3 个良性翻转击穿。定为校准试点（§10），门形态待地板数据决定；若良性翻转率
> 1%，接受单模型放宽并在论文注明——强行统一会毁掉其余 30 个模型的门槛可信度。

逐模型特记（协议/契约层面的关键钉死项）：

- **maskrcnn_v2 / keypointrcnn**：mask 的 paste_masks_in_image 含框坐标整数化舍入，是被
  测绘遗漏的离散步骤。契约二选一并写明：推荐产物输出 28×28 raw mask + 框，harness 用自己
  的 paste 实现对两侧同时贴回再比（离散舍入两侧同构抵消）；若 paste 属于要求转换的部分，
  必须加 H×W 侧 Gate B。
- **owlvit**：battery 变文本查询数（如 2/5/9，至少 1 个评测期才揭示）——形状固化的导出
  同 Q 全过、换 Q 即崩，单一 battery 测不到。
- **yoloworld / clipseg / groundingdino**：契约输入是 token ids / input_ids，文本塔必须在
  图束内——允许嵌入作输入 = 允许把最难转的文本塔留在束外 Python 里。
- **sam2_tiny**：钉死图像模式 + multimask_output=True；注意 SAM2 默认走 token-0 单掩码、
  仅 stability score 低于阈值时回退多掩码选优（dynamic_multimask_via_stability）——manifest
  需同时钉死该行为或实测确认 True 路径旁路它。
- **bigbird**：ref 钉死 attention_type='block_sparse'；battery 序列长度 ≥1024 多值——长
  序列上 full 与 block_sparse 数值可分，精度门天然兼任反作弊门（短序列回退区两者逐位一致，
  只测短序列会误判）。
- **rwkv**：契约强制 stateful `(ids_chunk, state_in) → (logits, state_out)`，harness 以
  随机 chunk 边界驱动并与 torch 整段比对；battery 长度非 2 幂（17/193/1023）。自一致性门
  改为"同一 artifact 两种 harness 侧分块调度（chunk=1 vs chunk=64）经 state 交接互比"，
  不要求产物具备整段模式（固定 chunk 图在 TFLite 可能是唯一可行导出）。
- **flan_t5**：三图契约（encoder / decoder / decoder_with_past）+ 自洽门：with-cache 逐步
  vs no-cache 整段重算——首步对、第二步起全错的 cache 接线错误只有这样才测得到。
- **switch_base8**：manifest 要求 router top-1 决策作辅助输出；电池设计期断言 8 专家覆盖；
  路由一致门是可直接观测决策面，允许硬门 + margin 豁免。
- **deberta_v3_small**：battery 多序列长度（64/128/384/512）——相对位置 log-bucket 索引
  算术被固化后，单长度 battery 完全测不出；附加导出体积 sanity（较权重膨胀 ~2× 提示索引
  常量固化）。

## 8. 反作弊闭环（已知向量 → 抓手）

| 作弊向量 | 抓手（独立多重） |
|---|---|
| fp16 假转（历史得手） | C1 dtype 普查（查文件）+ C2 数值指纹下界（查行为）——需同时骗过静态与动态两面 |
| YOLO-World 词表烘焙（历史得手） | C3 契约（text_ids 输入 + 束内文本塔）+ C8 差分（Δconv≈0 必挂）+ held-out 词表上 Gate B 比框——三条独立，任一存活即抓 |
| BigBird 切 original_full | ref 钉死 block_sparse + 长序列电池上两者数值差 ≫ 容差，主门自动判死（不查产物结构——图会被优化改写，纯行为判定） |
| RWKV 固定长度展开 | stateful 契约使展开图给不出 state_in/out 语义（结构性失败）+ 随机 chunk 等价 + 非 2 幂长度 |
| OWL-ViT 形状钉死 | 契约动态轴实跑 ≥3 值（含评测期揭示值） |
| 只导 backbone 骗 Gate A | Gate B 端到端槽 required，缺名死在 C3 |
| 框洪水 / 调阈值挤差异框 | 契约钉最大输出数 + precision 侧记账 + 豁免带只看 ref 侧分数 |
| 贴线优化 | Goodhart 监控：诊断落盘 margin 分布，指标大量紧贴阈值下方聚集 → 自动 flag 人工审计（正常转换误差远离阈值 3-6 个数量级，贴线本身是信号） |

**残余风险**（无完备解，如实声明）：蒸馏/查表混合的模仿模型、沙箱逃逸。行为门靠 1e-3 量级
紧度使模仿统计上不可行；权重指纹审计只做 best-effort 诊断不做硬门（算子融合合法摧毁指纹）。

## 9. 两项开赛前必须写死的政策决定

1. **fp16 赛道的语义**（当前最大未封作弊面）：fp16 存储 + 运行时全部上转 fp32 计算的产物
   能同时通过 C1（权重字节确是 fp16）、C2（权重舍入已使差异 > 下界）和全部数值门，却没有
   任何 fp16 计算收益——这是"不转 fp16"事件的直系变种。必须明文定义：仅要求存储（则明说
   并接受），还是要求计算路径 fp16（则需 per-format 手段：TFLite fp16 delegate 标志、图内
   dtype 检查、CoreML compute_precision 声明 + C2 双侧带）。这是政策决定不是比较器问题。
2. **基线公平性口径**（§4 已述）：主指标只看端到端 Gate B；Gate A/aux 槽/stateful 契约归
   反作弊门，同一 wrapper 对 agent 与 baseline 同等施加，Gate A 失败单列。

## 10. 校准与执行顺序（成败关键）

所有裸数字在地板校准前一律视为无效。顺序强制为：

1. **地板测量**：12 个 easy 模型是唯一在所有目标格式上存在"已知正确"转换产物的集合——用
   它们做每个 (format, precision) 的跨格式地板校准；hard 模型跑 torch 双后端（CPU vs
   MPS/CUDA）、双编译路径（eager vs compile）、fp64 参考的**全管线**地板（含 NMS/路由/哈希
   的良性翻转率）。统计口径与门一致：PASS 要求 5 trial 全过，地板就按 max-over-5-trials
   校准（否则多重检验系统性推高误报——现有 ±1/15 seed 波动的病根之一）。**reformer 为
   试点**：先实测 hash_seed 字段行为 + ref 双后端复现性 + 良性桶翻转率，再定门形态。
2. **冻结 tier 阈值**：阈值 = 地板最大观测 × 安全系数（聚合量 ×3，仲裁 ×2），校准工件与
   harness 版本一起存档。
3. **合成作弊 fixture 套件**：真的构造烘焙版 YOLO-World、full-attention BigBird、展开版
   RWKV、fp32 冒充 fp16 的产物，harness 任何改动必须让全部 fixture 保持红色。**必须先于
   匈牙利比较器上线**——matched_set 是全设计复杂度最高、最容易自带 bug 的部件（空集、全
   边缘带、重复框、代价并列），它的 bug 同时制造假阳假阴且会被读源码的 agent 定向利用。
4. **比较器上线**，每份 manifest 过元测试 + 独立评审。

配套工程保障：Canonicalizer 是"换格式零改动"承诺的单点故障（NHWC 转置、CoreML fp16 IO、
INT8 反量化任一静默 bug = 系统性假 FAIL 记在 agent 头上）——每个 RunnerAdapter 上线前用
3 个 pilot 模型的已知正确转换跑通全部比较器（金样自检），canonicalization 前后张量摘要
落盘可审计。另：easy 12 与 30 模型 registry 当前是两份未对齐的权威源（densenet201/resnet101
vs registry 的 densenet121/resnet18），需建唯一 registry（loader 调用 + weights hash），
校准工件绑定 registry 版本号。

## 11. 泛化与向后兼容

- **新模型** = 写 manifest + 电池规格 + 跑设计期 CI（C5 丰富度、C6 确定性），多数零代码；
  输出落在 3 比较器 × payload 之外才需要动库（判据见 §12）。
- **新格式**（TFLite/CoreML/TRT）= 新 RunnerAdapter + 该格式 tier 重校准，比较器一行不改。
  RT-DETR 的 GridSample TFLite 发散正是设计要抓的：同一比较器、TFLite tier 下跑，发散就是
  FAIL，不因 onnx2tf 自己的校验通过而豁免。
- **INT8** = 新 tier + C1 量化算子普查 + C2 改 SQNR 带（进场前须补官方判据调研与自校准；
  量化噪声成主导项后决策级一致门升为主门）。
- **KernelBench 向后兼容**：模型目录无 manifest 时生成默认 manifest（single_forward +
  单槽 dense）；dense 在预算置 0 时逐位复现旧判据——旧 harness 是新体系的退化情形，
  Phase 1 数字可复算（`legacy_strict: true` 开关）。

## 12. 何时才允许新增比较器（升级阶梯）

遇到不适配先走阶梯，新比较器是最后手段：

```
tier 参数 → manifest 结构参数 → Canonicalizer 变换（置换/排序/坐标系） → 新 payload 插件 → 新协议 → 新比较器
```

新增比较器须同时满足四条：(1) **不可归约**——存在无法用"规范化 + 现有度量"表达的输出
不变性，证据是可信正确转换在现有比较器下系统性假 FAIL；(2) **有假阴证据**——能构造语义
已破坏的转换通过全部现有门；(3) **非一次性**——需求出现在 ≥2 模型或一条已立项赛道；
(4) **参考自足**——仅凭 fp32 torch 参考输出可定义，不依赖标注数据/任务指标。每次新增附
校准数据 + 合成作弊 fixture，bump 库版本（判据改版前后成功率不可混报）。

## 13. 已知风险

1. **校准循环性**：hard 19 上不存在可信转换产物，噪声地板只能来自 torch 内部 ensemble，
   可能系统性低估跨 runtime 的合法内核差异（GridSample 实现级差异首当其冲）；fp64 仲裁兜
   一部分，其余需按算子建例外清单——手工且可被质疑。
2. **集合门阈值是全设计校准最弱环节**：拥挤场景下 1e-3 分数漂移经 NMS 放大可合法改变整个
   幸存框集合；豁免带窄了 fp16 档大面积假 FAIL、宽了成作弊面。fp16 检测组结论发布前必须
   公布校准数据。
3. **aux 槽契约的格式脆性**：让 4 种后端都干净暴露命名 aux 张量本身是转换难题的一部分
   （TFLite 重命名/融合张量、TRT 多 profile binding 易碎），可能把"比较层零改动"转嫁成
   "每格式 runner 大量特判"；且 aux 要求改变任务难度，必须写进发给两侧的任务契约。
4. **参考侧冻结随机性的破例**：reformer/bigbird 的 numpy 侧 RNG 不受 torch seed 控制，
   很可能需要参考侧 wrapper 代码——"新模型只写声明"在这几个模型上破例，提前登记。
5. **设计期断言会漂移**：专家覆盖、held-out 词表丰富度随权重/tokenizer 版本静默失效——
   全部做成随 bench 版本跑的 CI。

---

## 附录 A：31 模型输出签名与离散断点测绘

（完整测绘含"朴素比较为何失效"逐条论证；此处为判定方案速查。）

| 模型 | 输出签名 | 离散断点 | 判定方案 |
|---|---|---|---|
| maskrcnn_v2 | List[Dict]: boxes[N,4]/labels/scores/masks[N,1,H,W]，N 动态 0..100 | RPN topk、score 0.05、NMS 0.5、label 索引选 mask 通道、RoIAlign | G4 双门；mask 走 28×28 raw + harness 侧统一 paste |
| keypointrcnn | + keypoints[N,17,3]、kp_scores[N,17] | + 热图 argmax→整数像素 | G4 + keypoints payload（OKS/2px + 热图 margin 豁免） |
| owlvit | logits[B,576,Q]、boxes[B,576,4]，Q=查询数动态轴 | 本体 none；Q reshape 可被固化 | G3；battery 变 Q（2/5/9） |
| rtdetr_r18 | logits[B,300,80]、boxes[B,300,4]，无 NMS | encoder top-300 topk+gather；grid_sample 数值敏感 | G3；Gate A 用 encoder pre-topk 特征抓 GridSample 发散 |
| efficientdet_d0 | [B,100,6] NMS 后零填充固定行 | per-level topk、NMS、class 浮点存整数 | G4；先剥零行再 matched_set；Gate A 用裸模型 5 层头 |
| yoloworld_v2s | raw [1,4+K,8400]，K=词表动态轴 | raw 路径 none；文本嵌入可被烘焙 | G3 + 契约 text 输入 + C8 差分 + 多秘密词表 |
| groundingdino_t | logits[B,900,256]（token 对齐分）、boxes[B,900,4] | top-900 选择；文本 mask 构造；grid_sample | G3；dense 仅有效 token 列；battery 变 prompt 内容与长度 |
| sam2_tiny | masks[3,H,W]、iou_pred[3]、low_res_masks[3,256,256] | 0 阈值二值化；multimask=False 时 argmax(iou)；stability 回退 | G6；钉死 multimask=True；logits 按 token 槽 index dense |
| mask2former | class_q[B,100,81]、masks_q[B,100,H/4,W/4] 固定形状 | raw none；内部 0.5 阈值 attention mask 级联放大 | G3（raw 按 index dense）+ Gate B 后处理 mask_instance |
| birefnet | 末级 [B,1,1024,1024] logits | none；边界 logits 过零 rtol 病态 | G2；nRMSE 分母下限 + 分块 p99.9 + 二值 IoU 语义门 |
| clipseg | logits[Np,352,352]，Np=prompt 数 | none（FiLM 连续）；文本塔可烘焙 | G2 + 契约 text 输入 + C8 差分 |
| upernet_swin_t | logits[B,150,H,W]（~3.9e7 元素） | argmax 在后处理；roll 为长度依赖分支 | G2 + 像素 argmax 一致率（margin 豁免）；battery ≥2 分辨率 |
| longformer | last_hidden[B,L≤4096,768] | global mask 双分支（输入决定） | G2；battery 变长度 + 变 global mask 位置 |
| bigbird | last_hidden[B,L,768]，L 须 64 倍数 | 随机块索引（numpy 种子）、短序列回退 full | G2；battery ≥1024 多长度；ref 钉死 block_sparse |
| reformer | logits[B,L,V]，V≈320 | LSH 哈希→桶置换 ×num_hashes；种子敏感 | 校准试点†；nRMSE + per-token 预算 + top-1（margin 豁免） |
| switch_base8 | teacher-forced logits[B,T,32128] + router logits | 每 token× 每 MoE 层 top-1 路由 + scatter | G5 + 路由 discrete_agreement（可硬门）+ Gate A router logits |
| rwkv4_169m | logits[B,L,50277] + 每层 state | none（递归连续）；可被定长展开 | G5 state_carry_scan；变 chunk + state 交接；位置漂移斜率门 |
| flan_t5_small | teacher-forced logits；部署 2-3 图 + KV cache | 断点在图边界：cache 接线、相对位置偏置切片 | G5 双模式：整段 + 逐步 incremental（专抓 cache 错）；自洽门 |
| deberta_v3_small | last_hidden[B,L,768] | 相对位置 log-bucket 索引（长度依赖） | G2；battery 多长度；导出体积 sanity |
| 12 分类骨干 | logits[B,1000] 固定形状 | none | G1 全元素 + top-1 零翻转；googlenet 确认不带 aux 头、cvt 钉死分类头形态、vit/swin 输入分辨率写入契约 |

## 附录 B：设计过程与材料

本设计由 5-agent workflow 产出：1 个输出签名测绘员（31 模型逐一）、3 个独立设计师
（A 数值保真 / B 反作弊 / C 工程泛化）、1 个对抗批判者。综合裁决："骨架取 C（五层管线 +
3 比较器 × payload + tier 分离），宪法取 B（C1-C8 + 作弊 fixture + Goodhart 监控），数值
取 A（连续稠密判据 + margin 豁免公式 + 地板校准方法论 + fp64 仲裁），任何一家不能整体照抄。"
批判环节的主要否决：A 的豁免传播定理在注意力混合/检测级联下不成立（→ 校准预算）、B 的
"无近平分图"筛选不可行（→ margin 豁免 + 预算兜底）、C 的 1e4 模式开关与 INT8 qstep 单位
（→ 连续判据 + harness 侧步长）、三家全部裸阈值（→ 校准前一律无效）。
