# 精度匹配·简单分组方案（v1，配套代码在 `precision_matching/`）

> 定位：在现有 measure-don't-judge 流水线（四指标、随机输入、只记录不判定）上做**最小必要**
> 的分组扩展，让全部 33 个模型（31 + 新增 LaMa、MobileSAM）都能产出有意义的数字。
> 不做的事见 §7。完整的长期设计（反作弊、阈值校准、协议层）另见 `precision_matching_design.md`，
> 本方案是它的最小可用子集。

## 1. 分组结论（一句话）

四个指标（max_atol / max_rtol / nrmse / mismatch_ratio）**所有模型共用，不改**；
分组只发生在两处——**喂什么输入**（三组）和**怎么比输出**（两组），外加一个可选的
"第五个数"（决策一致率）辅助解读。

## 2. 输入分三组

| 输入组 | 生成方式 | 适用 |
|---|---|---|
| 随机小数（现状不变） | `[min, max]` 均匀采样，固定 seed | 图像分类 12 个 + 图像输入的分割/检测模型 |
| 随机整数（新增） | 声明整数 dtype 即生效，半开区间 `[min, max)`，如 token ids `[0, vocab_size)` | NLP 7 个的 token ids、CLIPSeg/OWL-ViT/GroundingDINO 的文本输入、SAM2/MobileSAM 的点标签 |
| 真实图片（新增） | 6 张固定 COCO val2017 图，逐 trial 轮换，双侧同图 | Mask R-CNN、Keypoint R-CNN（、EfficientDet 若用 bench 包装） |

**为什么检测必须用真实图片**：纯噪声图上检测器输出 0~2 个框（实测），两边都"啥也没检出"
一比完全一致——但挑框、去重这段最难转换的逻辑根本没被用上，等于没测。

**单 trial 模式（当前项目设置）**：整个评测只跑 1 个 trial 时，检测模型只会用到图目录里
按文件名排序的**第一张**图。先在有 torch 的机器上跑
`python3 -m precision_matching.check_images assets/real_images` 给 6 张图排名
（Mask R-CNN 高置信检出数 + Keypoint R-CNN 检出人数），把最好的一张留在目录里（或确保它
排序最前）。合格标准：≥10 个高置信框且 ≥2 个人——Keypoint R-CNN 只检人，没人的图对它等于
白卷。其余模型单 trial 影响很小（一次前向已是数万~千万个数值的逐值比较），个别模型数字
可疑时手动换 seed 重跑一次即可。

**随机整数的三个配套注意点**（不改代码，只是配置/加载时要做对）：

1. **flan_t5 / switch** 是 encoder-decoder，forward 还需要 `decoder_input_ids`——在 inputs
   里多声明一个整数输入即可；
2. **bigbird** 序列长度必须大于 (5+2·num_random_blocks)·block_size（默认配置即 >704），否则
   模型自动切回普通注意力，测的就不是稀疏路径；建议直接取 1024 并用 64 的倍数（非 64 倍数
   不会回退、只会被模型内部自动补齐，但取整倍数可避免 padding 引入的形状差异）；
3. **reformer** 加载参考模型时必须钉死 `config.hash_seed`，否则 LSH 随机投影每次前向重抽，
   torch 参考自己都不可复现（唯一需要动一行加载代码的模型）。

另外两个输入技巧：attention_mask 声明为整数 `[1, 2)` 即恒为全 1；LaMa 的挖洞 mask 用
`rect_mask` kind（随机矩形二值掩码，见 §5 spec 格式）。

## 3. 输出分两组

| 输出组 | 比法 | 适用 |
|---|---|---|
| 四指标直接比（现状不变） | 同形状张量逐一算四指标，跨张量取最差 | **30/33**——包括 RT-DETR、OWL-ViT、GroundingDINO、Mask2Former、YOLO-World（raw 输出都是固定形状张量！）、SAM2/MobileSAM（low_res_masks 固定形状）、LaMa（输出定形图像）、全部 NLP | 
| 先配对再比（新增） | 贪心 IoU 配对（同类别、IoU≥0.5、按分数降序），配对后记录：配对率、配对 IoU 均值/最小、分数差、两侧未配对的最高分 | **Mask R-CNN、Keypoint R-CNN**（torchvision 把后处理焊死在 forward 里，输出变长 list[dict]，绕不开）；EfficientDet 建议直接换裸模型输出 5 层特征头归入上一组 |

配对指标的读法：`unmatched_ref_max_score` 高（比如 0.8）= 转换后真的丢了一个高置信检测，
要人看；低（比如 0.06）= 只是阈值边缘抖动，正常。掩码（配对实例二值 IoU）和关键点
（配对 L2 像素距、可见性一致率）在配对后顺带记录。

检测输出的保存从 `np.save`（存不了 dict）换成 `save_detections`（npz）。

## 4. 可选的"第五个数"：答案变没变

四指标量"数值漂多远"，这个数记"离散决策翻没翻"，两边互相印证：数值难看 + 答案一致 =
多半是良性抖动；数值漂亮 + 答案翻了 = 恰恰要人工看的样本。

| 族 | 第五个数 | 函数 |
|---|---|---|
| 分类 | top-1 一致率 | `top1_agreement(ref, cand)` |
| 语义分割（upernet） | 像素类别一致率 | `top1_agreement(ref, cand, axis=1)` |
| LM logits（t5/rwkv/reformer/switch） | 逐位置 token top-1 一致率 | `top1_agreement(ref, cand)` |
| 二值掩码类（birefnet/clipseg/sam2/mobilesam） | 0.5 阈值化后 IoU | `binary_mask_iou(ref, cand)` |
| 检测 | 即 §3 的配对指标 | `match_detections(ref, cand)` |
| NLP encoder（longformer/bigbird/deberta） | 不需要（输出无决策环节） | — |

## 5. 代码清单与集成点

```
precision_matching/
  input_gen.py            # generate_inputs(specs, seed) —— 替换 run_inference 的输入生成段；
                          #   兼容现有 spec 格式 [[shape], min, max, dtype, layout]（+可选第6项 kind）
  real_images.py          # fetch_coco_images(dir) 一次性下载（http），评测期离线；
                          #   load_image_for_trial(dir, trial_i, shape) 逐 trial 确定性轮换
  check_images.py         # 单 trial 选图工具（需 torch，在 Mac 上跑一次）：给 6 张图按
                          #   Mask R-CNN 检出数 + 人数排名，选出唯一 trial 用哪张
  dense_metrics.py        # 四指标参考实现（公式与文档逐字一致）+ aggregate_worst；
                          #   另附 ref_nonfinite/cand_nonfinite 两个 NaN/Inf 计数诊断列（跨张量求和，
                          #   不参与四指标聚合）；非有限元素在 mismatch_ratio 中计为不匹配
  detection_matching.py   # match_detections / iou_matrix / save_detections / load_detections
  decision_agreement.py   # top1_agreement / binary_mask_iou
  tests/run_all.py        # 零依赖测试器：python3 precision_matching/tests/run_all.py（38 个用例）
```

集成三步：
1. `run_inference` 的输入生成段 → `generate_inputs`；检测三模型改走 `load_image_for_trial`
   （图片目录随 harness 打包，评测容器内不需要网络）；
2. `run_inference` 保存检测输出 → `save_detections`（npz）；
3. `run_evaluation` 对 maskrcnn/keypointrcnn 分支到 `match_detections`，其余照旧；
   `random_noise.json` 每模型多记一列第五个数（可选）。

## 6. 33 模型总表

| 模型 | 输入 | 额外输入声明 | 输出比较 | 第五个数 |
|---|---|---|---|---|
| 12 个分类骨干 | 随机小数 | — | 四指标 | top-1 一致率 |
| upernet_swin_t | 随机小数 | — | 四指标 | 像素类别一致率 |
| birefnet | 随机小数 | — | 四指标 | 掩码 IoU |
| **lama（新增）** | 随机小数 + rect_mask | 挖洞掩码 kind=rect_mask | 四指标（输出定形图像） | — |
| mask2former | 随机小数 | — | 四指标（两个 raw 头） | — |
| rtdetr | 随机小数 | — | 四指标（logits+boxes） | — |
| yoloworld_v2s | 随机小数 | 词表在导出前 set_classes 固化（简单版接受） | 四指标（raw head） | — |
| efficientdet_d0 | 随机小数 | 建议用裸模型（5 层头定形输出） | 四指标 | — |
| owlvit | 随机小数 + 随机整数 | input_ids `[0, vocab)`、attention_mask `[1,2)` | 四指标 | — |
| groundingdino | 随机小数 + 随机整数 | 同上 | 四指标 | — |
| clipseg | 随机小数 + 随机整数 | 同上 | 四指标 | 掩码 IoU |
| sam2_tiny | 随机小数 | 点坐标小数 `[0, 边长)`、点标签整数 `[0,2)` | 四指标（low_res_masks + iou_pred） | 掩码 IoU |
| **mobilesam（新增）** | 随机小数 | 同 sam2 | 四指标 | 掩码 IoU |
| maskrcnn_v2 | **真实图片** | — | **配对指标**（框+掩码） | （即配对指标） |
| keypointrcnn | **真实图片** | — | **配对指标**（框+关键点） | （即配对指标） |
| longformer / bigbird / deberta | 随机整数 | bigbird 长度取 1024（须 >704 才走稀疏路径） | 四指标 | — |
| reformer | 随机整数 | **钉 hash_seed** | 四指标 | token top-1 |
| flan_t5 / switch | 随机整数 | + decoder_input_ids | 四指标 | token top-1 |
| rwkv4_169m | 随机整数 | — | 四指标（logits + state） | token top-1 |

## 7. 明确不做的事（推迟项及其触发条件）

- **反作弊层**（dtype 普查、契约检查、敏感性差分）→ agent 进评测环时再上，最先补 dtype 普查
  （fp16 假转历史上真实发生过）；
- **自动 pass/fail 与阈值校准** → 需要自动判定时再做（目前人读数字）；
- **多长度电池、KV cache/状态接力、teacher forcing 协议** → 要测长度泛化/多图导出时再做
  （便宜的中间步：NLP 模型配两个序列长度各跑一遍，纯配置）；
- **YOLO-World 换词表、动态 query 数** 等条件轴探针 → 同上。
