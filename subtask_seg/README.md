# 一类：子任务标注（自动化）

**[A] 夹爪感知**（`../gripper`：SAM 2 跟踪左右夹爪 + 裁剪图开合分类）→ 抓取事件（闭合区间、开始张开的帧、空抓）
→ **[B] VLM 打分**（Qwen3-VL-8B 作为打分器，不生成文字：多选题 log-prob + 是非题 + 数数）
→ **[C] task sheet 语法 DP**（在候选边界上对齐）→ **[D] retry 规则** → 导出 + 自查。

人工输入（全部写进成本）：
- 训练夹爪模型的标注：框（每个训练片段几帧）+ 开合时间轴（只标切换帧）；
- 必做视频里的**锚点框**：每段几帧，只画两只夹爪的框，作为 SAM 2 跟踪的起点 / 纠正点；
- 没有对输出 JSON 做人工修正。

> 标 `TODO` 的地方需要跑完后填真实数字，不要留空也不要估。

## 运行（Oscar，conda 环境 `annotation`；目录结构）

```
video-annotation/
  data/                  必做视频 *.mp4、task_sheets.json、<id>_GT.json（手标参考，只用于评估）
  data/train/            训练用的非必做视频
  gripper/               夹爪跟踪 + 开合分类（见 gripper/README.md）
  checkpoints/           SAM 2 权重
  runs/gripper/          gripper 的全部产出（tracks/、pred/、cls/、states.json …）
  subtask_seg/           本目录
```

```bash
cd video-annotation
# 1) 夹爪：给视频画锚点框 → SAM 2 跟踪整段 → 开合分类
python -m gripper.prepare anchor --video 12_fold_four_towels.mp4 --frames 0 1250   # 然后在框工具里画框
python -m gripper.track video --videos 12_fold_four_towels.mp4
python -m gripper.state predict --model runs/gripper/cls/c1/weights/best.pt --videos 12_fold_four_towels.mp4

# 2) 子任务：一条命令一段视频
cd subtask_seg
python tools/measure_lag.py                                   # 量分类器滞后 -> ONSET_SHIFT
python annotate.py ../data/12_fold_four_towels.mp4 --onset-shift <N>   # 需要 GPU（VLM），答案缓存进 results/vlm/
python annotate.py ../data/12_fold_four_towels.mp4 --onset-shift <N> --no-vlm-compute   # 只改 DP 时：不用 GPU
python tools/eval_gt.py results ../data                       # 和手标参考比较
python tests/synthetic_check.py                               # 逻辑单测（无需 GPU / 视频）
```

`run_all.sh` / `slurm/annotate.sbatch` 批量跑所有已有夹爪预测的视频。VLM 权重路径在 `pipeline/config.py`
的 `VLM_MODEL`（或 `--vlm-model`）。

输出：`results/<id>.json`（提交格式）、`results/vlm/<id>.json`（VLM 全部答案的缓存）、
`results/debug/<id>_events.json`（抓取事件、候选边界、每段的 VLM 分数和结束类型、跳过的 slot、retry 理由、自查问题）、
`results/<id>.eval.csv`（和参考比较的逐行误差）。

## 思路

| 模块 | 文件 | 解决什么 |
|---|---|---|
| 夹爪位置 | `../gripper/track.py` | SAM 2（base_plus）从人工锚点框出发跟踪左右夹爪；50 帧一个窗口，用上一窗口的掩码框重新提示；路过锚点时用人工框纠正。每帧都有框；夹爪被挡住时沿用上一帧的框并标记。左右身份由跟踪保证，不会互换 |
| 夹爪开合 | `../gripper/state.py` | YOLO11s-cls 对夹爪的方形裁剪图（原 1280×720 帧上裁，256²）分 open / closed / unknown → `p_closed`；unknown、被挡住的帧记缺失 |
| 抓取事件 | `pipeline/events.py` | `1 − p_closed` 中值滤波 + 滞回阈值 → 闭合区间；“开始张开的第一帧” = 从重新张开处往回走，找第一帧高出闭合平台 `ONSET_DELTA` 的位置，再减去 `ONSET_SHIFT`；空抓 = 短闭合 + 物体没动 + 同一夹爪 3 s 内在同处再抓 |
| VLM 打分 | `pipeline/vlm.py` | 每 1 s 取一个 2 s 片段（6 帧，宽 960），问“机器人在做 task sheet 的哪一行”，选项 = 该任务全部行，读各选项字母的 log-prob（选项字母每个片段随机打乱，DP 里再减掉每个选项在整段视频上的均值，去掉字母 / 选项偏好）；条件行（`only if …`）在开头几帧问是非题；按件计数的物体问“有几个”，读数字 0–9 的概率；空抓候选问“夹爪有没有拿起 / 移动物体” |
| 语法 DP | `pipeline/tasksheet.py` `pipeline/align.py` | task sheet 展开成按顺序的 slot（固定次数 / 条件行 / 按件计数 / 重复轮次 / 最后的 “Last frame” 行）。候选边界 = 松手帧（[A]）、夹爪从静止开始移动的帧、1 s 网格兜底；以松手结束的行优先落在松手帧，其他行优先落在开始移动的帧，网格要额外代价。DP 代价 = −VLM log-prob（中心化后在段内求和）+ 结束类型代价 + 跳过 / 多用 slot 的代价（必选行的跳过代价按 VLM 数数的概率缩放，条件行按是非题概率）。“Last frame” 行固定结束在最后一帧 |
| retry | `pipeline/retry.py` | 空抓：闭合 ≥ 0.4 s、几乎没动、同一夹爪 3 s 内在同处再抓，且 VLM p(拿起) < 0.5；放下后在原处再抓（餐具等 task notes 说算 retry 的行）也算；叠 / 拉直毛巾时的松开再捏不算。slot 内不是最后一次的失败 → suboptimal，recovery_t = 最后一次伸手的开始 |
| 输出 | `pipeline/export.py` | 首尾相接的 subtask、step 区间、status / recovery_t / note |
| 自查 | `pipeline/validate.py` | 规范里的 5 项检查 + 格式检查，每段运行时打印 |
| 评估 | `tools/eval_gt.py` `tools/measure_lag.py` | 和手标参考比边界 / 名字 / 次数 / status；用开合时间轴量分类器滞后 |

**边界怎么定到 ±0.5 s。** 大多数 subtask 以“夹爪开始张开的第一帧”结束，这是夹爪上的几何事件，所以直接从每只
夹爪的开合曲线读，不靠时序模型猜。分类器按单帧外观判断，相对真正开始张开有固定滞后：开合时间轴的标注规则是
“切换点 = 手指开始分开的第一帧”，`tools/measure_lag.py` 用分类器验证集片段上的这些切换点量出滞后，写进
`--onset-shift`。两只夹爪都松开时取 slot 内最晚的松开。不松手的行用“夹爪离开物体框并持续 5 帧”。

**怎么对上名字和次数。** 名字一律取 task sheet 原文，DP 只决定“哪一行、从哪帧到哪帧”。夹爪事件只知道“什么时候
松手”，不知道“在做哪一行”；VLM 反过来看得懂在做什么，但时间分辨率只有 1 s。所以边界只允许落在夹爪事件（或兜底网格）
上，行的身份由 VLM 分数决定，两者在 DP 里合并。次数：固定 count 直接展开；按件计数用 VLM 数数的众数展开，再允许
少用 / 多用（代价按数数概率）；条件行的用 / 不用代价按是非题概率。

**怎么发现 retry。** 见上表 retry 行。规则只看 [A] 的抓取事件，VLM 只做一次是非确认，避免分类器抖动造成的短闭合被当成空抓。

**为什么这样选。** 规范几乎都锚定在夹爪开合上，所以把精度押在“夹爪在哪 + 开没开”上。最早用 YOLO 检测器逐帧找
夹爪，换场景就找不到（r1 在 12 号视频上约 70% 的帧没框）；改成 SAM 2 从少量锚点框跟踪后每帧都有框，开合判断
变成对小图分类。task sheet 已经给了顺序和次数，“分段 + 命名”退化成约束很强的对齐问题，用 DP 精确求解。

**试过但没成功的方法。**
- YOLO11 检测器直接检测 open / closed 夹爪（66 帧训练）：val mAP50 0.65，换到 12 号视频大面积漏检。
- YOLOE 开放词表“open / closed robot gripper”：分不出开合；只用来给训练帧预标夹爪位置。
- YOLO-World + ByteTrack 检测物体，用物体位移判断拿起 / 关联 task sheet 行：26 号上几乎所有餐具都被叫成
  teaspoon，物体类别对不上行；不再使用（`perception.detect_objects` 保留作实验）。
- 只用抓取事件做分段 DP（没有 VLM）：12 号的 24 次抓取被整团分进第一个 slot，±25 帧内 0%；松手帧本身是准的
  （8 个松手误差 −51…16 帧），错在“哪段属于哪行”，因此加了 [B] VLM 打分。
- 指尖关键点（YOLO-pose）：俯拍时指尖经常被挡，松手那一刻恰好读不到开口度。
- 其他 TODO（如实写）。

## 怎么知道准不准

1. `tests/synthetic_check.py`：构造已知事件的假感知数据，断言边界误差 ≤ 3 帧、retry 被识别、次数 / step 正确。
2. 每段运行自动跑 `validate.py`（规范 5 项）。
3. 夹爪跟踪：留出的人工框与跟踪框的 IoU、中心偏移（`python -m gripper.track report`）：TODO。
4. 开合分类：按片段划分的验证集 top-1 TODO；13、16 号（训练时未见过的白色臂场景）TODO。
5. 分类器滞后（`measure_lag.py`，验证集片段）：TODO 帧。
6. 手标参考（`data/<id>_GT.json`，task player 逐帧标）。**说明：12、26 号的参考被当作开发集，用来调 DP 代价 /
   阈值和发现错误，所以下面的数字偏乐观；12、26 号从未进入夹爪检测 / 分类器的训练集。** 只有其他视频才是真正的留出评估。
   - 12_fold_four_towels：TODO 个边界，median |e| TODO 帧，±25 帧内 TODO%
   - 26_sort_cutlery_drawer_insert：TODO
   - 人工标参考用时：TODO 分钟

## 已知错误与风险

- TODO：跑完后按“视频名 + 时间 + 现象”逐条写（`eval.csv` 里误差 > 25 帧的行 + validate 的 CHECK）。
- 26 号场景里只有一把 dinner spoon（task sheet 写 2 把，人工确认）；VLM 数数若数错，DP 有把别的抓取凑成第二把
  勺子的风险。
- VLM 每 1 s 一个片段、6 帧：短于 1 s 的行身份靠相邻片段，可能被并进邻行。
- 先做后面的 step（顺序和 task sheet 不同）时，DP 不能乱序，只会跳过 / 错配（已知局限）。
- 白色臂（YAM）上 SAM 2 的掩码常连带相机支架 / 手腕，框偏大；夹爪被完全挡住时框沿用上一帧。
- 压、擦、推等不松手的行，边界落在“开始移动”的候选或 1 s 网格上，精度低于松手行。

## 成本

- 硬件：Oscar GPU（TODO 型号）；SAM 2 跟踪约 10 fps（base_plus）；Qwen3-VL-8B 每段视频约 TODO 次前向（每秒 1 次 + 少量是非题）
- 每段运行时间：TODO
- API 费用：0
- 人工：训练帧框 TODO 帧 / TODO 分钟；开合时间轴 24 个单元（12000 帧）/ TODO 分钟；必做视频锚点框 TODO 帧 /
  TODO 分钟；输出 JSON 人工修正 0

## 扩展到 1 万小时

- 人工锚点框是第一个瓶颈：用已有的锚点 + 跟踪结果训练一个夹爪检测器，只在跟踪告警处（框突变、被挡）自动重新提示，
  人工只复核告警。
- SAM 2 逐帧跟踪是算力瓶颈（约 10 fps）：降到 25 fps 跟踪 + 插值，或只在闭合区间附近逐帧跑。
- 开合分类器对新机器人 / 新相机要补时间轴标注；有夹爪关节数据时可直接生成标签。
- VLM 是每秒一次前向，1 万小时 ≈ 3600 万次：可换小模型 / 降帧数，或只在 DP 不确定的区间问。
- DP 代价和阈值现在手调；有了更多参考后在验证集上网格搜索，并按 `align_cost` 优先把最可疑的视频送人工复核。
