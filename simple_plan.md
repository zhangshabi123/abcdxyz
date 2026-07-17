# 精度匹配·简单分组方案（v2，配套代码在 `precision_matching/`）

> 定位：在现有 measure-don't-judge 流水线（四指标、随机输入、只记录不判定）上做**最小必要**
> 的分组扩展，让全部 33 个模型（31 + 新增 LaMa、MobileSAM）都能产出有意义的数字。
> 不做的事见 §6。完整的长期设计（反作弊、阈值校准、协议层）另见 `precision_matching_design.md`，
> 本方案是它的最小可用子集。
>
> v2 变更：定稿为「value 级 × 任务级」二维设计；任务级验证从"可选的第五个数"升级为
> 按族明确定义的一列；agent 可见性/防作弊问题整体推迟（§6）。

## 1. 设计结论（两句话）

**value 级验证**（逐值比较，四指标）：所有模型都做，输入用"能填满输出张量的最便宜的"——
除 R-CNN 系外全部是随机输入；R-CNN 系因为"没发现就不开口"的输出规则，噪声下无值可比，
用一张真实图。

**任务级验证**（答案变没变）：只在"有离散答案可翻"的地方做，输入一律用真实样本——
检测比配对后的框集合，语言模型比 next-token 预测；分类不做（噪声图上的 top-1 没有任务语义），
encoder-only NLP 没有答案可翻（详见 §3）。

| 模型族 | value 级 | 任务级 |
|---|---|---|
| 图像分类 12 个 | 随机噪声 | 不做 |
| NLP 7 个 | 随机整数（token ids） | 一个真实文本样本 |
| R-CNN 系 2 个 | 一张真实图（配对后比值） | 同一张图（配对率/IoU） |
| 其他检测 | 随机噪声（raw 定形张量） | 一张真实图（后处理后配对） |
| 分割系 / LaMa | 随机噪声（+点提示/rect_mask） | 可选：同一张真实图记 mask IoU |

## 2. value 级验证：输入怎么生成

| 输入组 | 生成方式 | 适用 |
|---|---|---|
| 随机小数（现状不变） | `[min, max]` 均匀采样，固定 seed | 图像分类 + 图像输入的分割/检测模型 |
| 随机整数（新增） | 声明整数 dtype 即生效，半开区间 `[min, max)`，如 token ids `[0, vocab_size)` | NLP 的 token ids、CLIPSeg/OWL-ViT/GroundingDINO 的文本输入、SAM2/MobileSAM 的点标签 |
| 真实图片（新增） | 图目录按文件名排序取图，双侧同图 | 仅 Mask R-CNN、Keypoint R-CNN（EfficientDet 用裸模型即可留在噪声组） |

**为什么 R-CNN 必须用真实图片**：它的输出规则是"发现了东西才汇报"。纯噪声图上两边都交
白卷（实测 0~2 个框），白卷对白卷完全一致，但脑内几千万次运算一个数都没看到；偶尔冒出的
一两个框全是贴着 0.05 分数线的骑墙货，合法的浮点抖动就能让一边有框一边没框。真实图让它
说出几十行远离门槛的数值，配对之后照样是逐值比较。

**单 trial 模式（当前项目设置）**：只跑 1 个 trial 时检测模型只用到排序第一的那张图。先在
有 torch 的机器上跑 `python3 -m precision_matching.check_images assets/real_images` 给图排名
（Mask R-CNN 高置信检出数 + Keypoint R-CNN 人数），把最好的一张放在最前。合格标准：≥10 个
高置信框且 ≥2 个人（Keypoint R-CNN 只检人，没人的图对它等于白卷）。其余模型单 trial 影响
很小（一次前向已是数万~千万个数值的逐值比较），个别模型数字可疑时手动换 seed 重跑即可。

**随机整数的三个配套注意点**（配置/加载时要做对）：

1. **flan_t5 / switch** 是 encoder-decoder，forward 还需要 `decoder_input_ids`——inputs 里
   多声明一个整数输入即可；
2. **bigbird** 序列长度必须大于 (5+2·num_random_blocks)·block_size（默认配置即 >704），
   否则自动切回普通注意力，测的就不是稀疏路径；建议取 1024 且用 64 的倍数（非 64 倍数不会
   回退、只会被自动补齐，取整倍数可避免 padding 引入的形状差异）；
3. **reformer** 加载参考模型时必须钉死 `config.hash_seed`，否则 LSH 随机投影每次前向重抽，
   torch 参考自己都不可复现（唯一需要动一行加载代码的模型）。

另外两个输入技巧：attention_mask 声明为整数 `[1, 2)` 即恒为全 1；LaMa 的挖洞 mask 用
`rect_mask` kind（随机矩形二值掩码）。

**value 级怎么比**：同形状张量逐一算四指标（max_atol / max_rtol / nrmse / mismatch_ratio，
公式与现有文档一致），跨张量取最差。R-CNN 例外：先贪心 IoU 配对（同类别、IoU≥0.5、按分数
降序），配对后比框坐标/分数——配对是对齐手段，比的仍然是值。检测输出的保存从 `np.save`
（存不了 dict）换成 `save_detections`（npz）。

## 3. 任务级验证：按族定义

四指标量"数值漂多远"，任务级记"答案翻没翻"，两列互相印证：数值难看 + 答案一致 = 多半是
良性离散抖动；数值漂亮 + 答案翻了 = 恰恰要人工看的样本。

| 族 | 任务级输入 | 任务级指标 | 说明 |
|---|---|---|---|
| R-CNN 系 | 与 value 级**同一张**真实图 | 配对率、配对 IoU 均值/最小、检出数差、两侧未配对最高分；掩码二值 IoU / 关键点 L2 | 与 value 级是同一次前向的两层读数，零额外成本 |
| 其他检测 | 一张真实图（与 R-CNN 共用同一张） | **harness 侧后处理**（阈值/top-k/NMS，同一份代码跑两边的 raw 输出）后 `match_detections` | 这次前向的 raw 张量顺手也记四指标——免费覆盖噪声探不到的高激活数值区间 |
| NLP：LM/seq2seq（flan_t5、switch、rwkv、reformer） | 一段真实长文本（各自 tokenizer 处理，双侧同 ids） | 逐位置 next-token top-1 一致率（`top1_agreement`） | 真·任务级："这段文章上两边对下一个词的预测 99.x% 位置一致" |
| NLP：encoder-only（longformer、bigbird、deberta） | 同上 | （可选）句向量余弦 | **没有离散答案可翻**——本质是在真实分布上的第二个 value 级点，别当任务语义读 |
| 分类 | — | 不做 | 噪声图上的 top-1 没有任务语义；四指标已全覆盖 1000 个 logits |
| 分割系（可选） | 同一张真实图 | `binary_mask_iou` / 像素 argmax 一致率 | 不做也不破坏设计 |

两个必守的细节：

1. **真实文本要选长的**（≥1024 token），否则 bigbird 掉回普通注意力——随机整数 trial 保证了
   长度，真实样本 trial 同样要保证；
2. **开放词汇检测器（OWL-ViT / GroundingDINO）的任务级 trial 文本必须用真实类名**（如 COCO
   80 类，tokenize 后双侧同输入）——喂随机 token 等于在真实图里找乱码词，检不出东西，任务级
   退化成白卷对白卷。YOLO-World 词表在导出前已固化，不受影响。

前向次数记账（单 trial 口径）：分类/分割 1 次；R-CNN 1 次（两级同源）；其他检测 2 次
（噪声 value + 真实图任务级）；NLP 2 次（随机整数 value + 真实文本任务级）。

## 4. 代码清单与集成点

```
precision_matching/
  input_gen.py            # generate_inputs(specs, seed) —— 替换 run_inference 的输入生成段；
                          #   兼容现有 spec 格式 [[shape], min, max, dtype, layout]（+可选第6项 kind）
  real_images.py          # fetch_coco_images(dir) 一次性下载（http），评测期离线；
                          #   load_image_for_trial(dir, trial_i, shape) 确定性取图
  check_images.py         # 选图工具（需 torch，在 Mac 上跑一次）：按 Mask R-CNN 检出数 +
                          #   人数排名，决定唯一 trial 用哪张
  dense_metrics.py        # 四指标参考实现（公式与文档逐字一致）+ aggregate_worst；
                          #   另附 ref_nonfinite/cand_nonfinite 两个 NaN/Inf 计数诊断列（跨张量求和，
                          #   不参与四指标聚合）；非有限元素在 mismatch_ratio 中计为不匹配
  detection_matching.py   # match_detections / iou_matrix / save_detections / load_detections
  decision_agreement.py   # top1_agreement / binary_mask_iou —— 任务级指标的实现
  tests/run_all.py        # 零依赖测试器：python3 precision_matching/tests/run_all.py（38 个用例）
```

集成四步：
1. `run_inference` 的输入生成段 → `generate_inputs`；R-CNN 与任务级 trial 改走
   `load_image_for_trial`（图片目录随 harness 打包，评测容器内不需要网络）；
2. `run_inference` 保存检测输出 → `save_detections`（npz）；
3. `run_evaluation` 对 maskrcnn/keypointrcnn 分支到 `match_detections`，其余照旧走四指标；
4. 任务级：其他检测模型在 harness 侧对两边 raw 输出跑同一份后处理再 `match_detections`；
   NLP 对真实文本 logits 记 `top1_agreement`。结果与 value 级分列记录，不混合聚合。

## 5. 33 模型总表

| 模型 | value 级输入 | 额外输入声明 | value 级比较 | 任务级 |
|---|---|---|---|---|
| 12 个分类骨干 | 随机小数 | — | 四指标 | 不做 |
| upernet_swin_t | 随机小数 | — | 四指标 | 可选：像素类别一致率（真实图） |
| birefnet | 随机小数 | — | 四指标 | 可选：掩码 IoU（真实图） |
| lama | 随机小数 + rect_mask | 挖洞掩码 kind=rect_mask | 四指标（输出定形图像） | 不做 |
| mask2former | 随机小数 | — | 四指标（两个 raw 头） | 可选：实例后处理配对（真实图） |
| rtdetr | 随机小数 | — | 四指标（logits+boxes） | 真实图 + 后处理配对 |
| yoloworld_v2s | 随机小数 | 词表导出前 set_classes 固化 | 四指标（raw head） | 真实图 + 后处理配对 |
| efficientdet_d0 | 随机小数 | 建议用裸模型（5 层头定形输出） | 四指标 | 真实图 + 后处理配对 |
| owlvit | 随机小数 + 随机整数 | input_ids `[0, vocab)`、attention_mask `[1,2)` | 四指标 | 真实图 + **真实类名** + 后处理配对 |
| groundingdino | 随机小数 + 随机整数 | 同上 | 四指标 | 真实图 + **真实类名** + 后处理配对 |
| clipseg | 随机小数 + 随机整数 | 同上 | 四指标 | 可选：掩码 IoU（真实图 + 真实类名） |
| sam2_tiny | 随机小数 | 点坐标小数 `[0, 边长)`、点标签整数 `[0,2)` | 四指标（low_res_masks + iou_pred） | 可选：掩码 IoU（真实图） |
| mobilesam | 随机小数 | 同 sam2 | 四指标 | 可选：掩码 IoU（真实图） |
| maskrcnn_v2 | **真实图片** | — | 配对后比值（框+掩码） | 同一张图：配对率/IoU/检出数差 |
| keypointrcnn | **真实图片** | — | 配对后比值（框+关键点） | 同一张图：配对率/关键点 L2 |
| longformer / bigbird / deberta | 随机整数 | bigbird 长度取 1024（须 >704 才走稀疏路径） | 四指标 | 真实长文本（第二 value 级点，可选句向量余弦） |
| reformer | 随机整数 | **钉 hash_seed** | 四指标 | 真实长文本 + token top-1 一致率 |
| flan_t5 / switch | 随机整数 | + decoder_input_ids | 四指标 | 真实长文本 + token top-1 一致率 |
| rwkv4_169m | 随机整数 | — | 四指标（logits + state） | 真实长文本 + token top-1 一致率 |

## 6. 明确不做的事（推迟项及其触发条件）

- **agent 信息隔离与防作弊**（用户决定整体推迟）：agent 可读评测脚本、评测输入保密、发同分布
  debug 样本（评测 A.jpg 保密、给 agent B.jpg）的策略本身已定，但实施细节推迟。届时第一件事：
  **噪声模型的秘密是 seed**——当前 seed=trial 序号写在可读脚本里，评测时须改为外部注入的
  秘密 seed（一次评测内 ref/cand 共用即可，无须提前公开）；同时 debug 图要按 A 图同样的
  丰富度标准挑，A 图避开著名 COCO 图；
- **反作弊层其余部分**（dtype 普查、契约检查、敏感性差分）→ agent 进评测环时再上，最先补
  dtype 普查（fp16 假转历史上真实发生过）；
- **自动 pass/fail 与阈值校准** → 需要自动判定时再做（目前人读数字）；
- **多长度电池、KV cache/状态接力、teacher forcing 协议** → 要测长度泛化/多图导出时再做
  （便宜的中间步：NLP 模型配两个序列长度各跑一遍，纯配置）；
- **YOLO-World 换词表、动态 query 数** 等条件轴探针 → 同上。
