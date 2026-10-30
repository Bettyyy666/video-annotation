# video-annotation

给双臂遥操作机器人视频自动标注 **subtask 分段**（名字、次数、起止帧、status / retry），输出格式和 `data/<id>_GT.json` 一致。
核心的辅助部分是夹爪定位 + 开合判断（`gripper/`）。

> **现状**：夹爪“什么时候松手”已经能做到帧级（训练片段上 median |误差| 1 帧，val 7 帧）。但“哪一段属于 task sheet
> 的哪一行”还没做好：12、26 号两段视频的 16 个边界里，只有 1 个落在 ±0.5 s 内（median |误差| 356 帧 ≈ 7 s）。
> 瓶颈在 VLM 对行的识别（每秒片段的 top-1 行只有约 30% 是对的），不在边界定位本身。详见第 5 节。

---

## 1. 目录和要下载的模型

```
video-annotation/
  data/                     必做视频 12、26 的 *.mp4、task_sheets.json、<id>_GT.json（手标参考，只用于评估）
  data/train/               非必做视频（09 11 13 14 16 21 27），只用来训练夹爪模型
  gripper/                  [A] 夹爪：SAM 2 跟踪 + 开合分类 + 标注工具 label_tool（详见 gripper/README.md）
  subtask_seg/              [B][C][D] 子任务分段：VLM 打分 + task sheet DP + retry（详见 subtask_seg/README.md）
  checkpoints/              SAM 2 权重（下载后放这里，不上传）
  task_player_assets/       手动标 GT 用的逐帧播放器（在 Mac 上跑）
  runs/gripper/             gripper 的全部产出（标注、tracks/、pred/、cls/ 模型）
  runs/exp1/                早期 Qwen 实验的输出
  Qwen_exp1.py yolo_exp1.py 早期实验脚本（见第 4 节）
  run_c1.sbatch             一次跑完：训练 c1 → 量滞后 → 跟踪 12、26 → 出 JSON → 和 GT 比较
  environment.yml           conda env `annotation` 的依赖
```

`gripper/common.py` 和 `subtask_seg/pipeline/config.py` 里写死了根目录 `/users/rzhou52/data/users/rzhou52`，换机器时要改这两处。

| 模型 | 用途 | 放在哪里 | 来源 |
|---|---|---|---|
| SAM 2.1 hiera **base_plus**（必需） | 跟踪左右夹爪 | `video-annotation/checkpoints/sam2.1_hiera_base_plus.pt` | `https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt`；代码用 `pip install "git+https://github.com/facebookresearch/sam2.git"` 安装 |
| YOLO11s-cls（必需） | 开合分类器的预训练权重 | `/users/rzhou52/data/users/rzhou52/YOLO11/yolo11s-cls.pt` | ultralytics 官方 |
| 开合分类器 c1（训练得到） | 每帧判断夹爪 open / closed / unknown | `runs/gripper/cls/c1/weights/best.pt` | `python -m gripper.state train --name c1` |
| Qwen3-VL-8B-Instruct（必需，17 GB） | [B] 给每秒片段打分 | `/users/rzhou52/data/users/rzhou52/Qwen3-VL-8B-Instruct/` | HuggingFace `Qwen/Qwen3-VL-8B-Instruct` |
| YOLOE-11l-seg + `mobileclip_blt.ts`（可选） | 只用来给新片段预标夹爪框 | `/users/rzhou52/data/users/rzhou52/YOLO11/yoloe/`（两个文件放同一目录） | ultralytics 官方 |
| YOLO-World v2 x + CLIP（可选） | `annotate.py --objects` 的物体检测，默认不用 | `subtask_seg/yolov8x-worldv2.pt`，CLIP 自动缓存 | `cd subtask_seg && python tools/prefetch.py`（在登录节点跑，需要联网） |

环境：conda env `annotation`，依赖写在 `environment.yml`（Python 3.11、torch 2.8 + cu128、sam2、ultralytics、transformers）。在 `video-annotation/` 下一条命令安装：

```bash
conda env create -f environment.yml && conda run -n annotation env SAM2_BUILD_CUDA=0 pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4"
conda activate annotation
```

sam2 要在 torch 装好之后再编译，所以放在第二步。

---

## 2. 怎么运行

### 2.1 自动流程（GPU 节点）

```bash
cd /users/rzhou52/data/users/rzhou52/video-annotation
sbatch run_c1.sbatch                                  # 全流程，日志写到 run_c1_<jobid>.log
# 或者在已经 salloc 到的 GPU shell 里（annotation 已激活）：
nohup bash run_c1.sbatch > run_c1.log 2>&1 &

# c1 已经训好时，只跑 12 / 26（跟踪 → 分类 → 分段 → 评估），约 35 分钟：
cd subtask_seg && sbatch slurm/run_dev.sbatch -7      # 或在 GPU shell 里：bash run_dev.sh -7
# 只改了 DP / retry 代码时不需要 GPU（用缓存的 VLM 答案）：
python annotate.py ../data/12_fold_four_towels.mp4 --onset-shift -7 --no-vlm-compute
python tools/eval_gt.py results ../data               # 和手标参考比较
python tests/synthetic_check.py                       # 逻辑单测，不需要 GPU / 视频，应输出 OK: ...
```

输出：
- `subtask_seg/results/<id>.json`：提交格式
- `results/<id>.eval.csv`：逐行误差
- `results/debug/<id>_events.json`：抓取事件、候选边界、VLM 分数、跳过的 slot、自查问题
- `results/vlm/<id>.json`：VLM 答案缓存

### 2.2 夹爪标注工具 label_tool（框 + 开合时间轴）

在 GPU 节点（或任何能读 `runs/gripper/` 的节点）上运行：

```bash
python -m gripper.label_tool --port 8765
#   http://localhost:8765/        框工具：给每帧画 L / R 夹爪框（训练片段，以及必做视频的锚点）
#   http://localhost:8765/state   开合时间轴：一个单元 = 一只夹爪 × 一个 10 s 片段，只在状态切换的那一帧按 1/2/3
```

这个服务只监听计算节点自己的 127.0.0.1，浏览器要通过 SSH 隧道才能访问。做法是在 **Mac 终端**里经 vscode 节点跳到计算节点。
`ssh.ccv.brown.edu` 这个网关解析不了 `gpuXXXX`，所以不能直接用它做 `-J`；在我的环境里，VS Code 的 Ports 面板也加不上端口。

```bash
ssh -N -L 8765:localhost:8765 -J ccv-vscode-node rzhou52@gpu2113     # gpu2113 换成 squeue --me 里显示的节点
```

`ccv-vscode-node` 是 Mac 上 `~/.ssh/config` 里 VS Code 用的那个 Host（`ProxyCommand ssh -q -W %h:%p jump-box`）。
快捷键和标注规则见 `gripper/README.md`。前后常用的命令：

```bash
python -m gripper.prepare prelabel                    # YOLOE 预标框（默认权重路径已写好，不需要设 $YOLO）
python -m gripper.track segments                      # 跟踪所有训练片段，并用其余的人工框检查 IoU
python -m gripper.track report                        # 跟踪框 vs 人工框：ok / extent（范围不同）/ off（跟错地方）
python -m gripper.state export && python -m gripper.state train --name c1
python -m gripper.state predict --model runs/gripper/cls/c1/weights/best.pt [--videos ... --vis]
python -m gripper.prepare anchor --video 12_fold_four_towels.mp4 --frames 0    # 给必做视频加锚点帧
```

### 2.3 task player：手动标 subtask GT（现在就能用）

这是一个逐帧播放器，在 **Mac 本机**上运行，不需要 GPU 或模型。它用来标 `data/<id>_GT.json` 这样的参考答案，也可以直接用来人工标新视频。

```bash
# 在 Mac 上 task_player_assets/ 所在的那份 video-annotation 目录里：
双击 task_player_assets/打开播放器.command        # 相当于 python3 serve.py，会打开 http://127.0.0.1:8766/index.html
python3 task_player_assets/serve.py --port 8767   # 端口被占用时换一个
```

- 播放的是逐帧解码出来的 JPEG，帧号就是原视频的帧号，不依赖浏览器的 seek。
- `→ / ←` 跳 10 帧，`Shift+→ / ←` 跳 1 帧，`Space` 自动前进，输入帧号后回车跳转。
- 展开“记录 task ground truth”，选好 subtask，停在结束帧按 `B`；按 `R` 记录 recovery_t（状态会自动设为 suboptimal）。下一段自动接在上一段结尾，steps 自动汇总。
- 用“导入 / 切换视频”可以直接导入新的 50 fps 视频（≤ 2 GB），每个视频按内容指纹分别保存草稿。
- “导出标注 JSON”会先做完整性校验，再输出 `{episode, steps, subtasks}`；草稿可以随时下载。

注意：`serve.py` 默认读取 `../players/12_fold_four_towels/player-data.json`，这些帧数据由 `task_player.py` 生成。
`task_player.py` 和 `players/` 只在 Mac 那份目录里，**集群上的这份没有**；要在集群上用，得先把它们同步过来。详见 `task_player_assets/README.md`。

---

## 3. 思路：pipeline 由哪些部分组成

```
[A] 夹爪感知 (gripper/)                [B] VLM 打分 (Qwen3-VL-8B)            [C] task sheet 语法 DP         [D] retry 规则
人工锚点框 → SAM 2 跟踪 L/R 夹爪   ┐   每 1 s 取一个 2 s 片段，问“在做哪一行”  ┐                             ┐
→ 裁剪图 YOLO11-cls 开合 → p_closed ├─► 候选边界（松手帧 / 开始移动 / 1 s 网格）├─► 在候选边界上对齐行序列 ──► ├─► 空抓 / 放下重抓 → suboptimal
→ 闭合区间、“开始张开的第一帧”    ┘   + 条件行是非题、按件数数、空抓确认       ┘   名字、次数取自 task sheet   ┘   + recovery_t → 导出 + 自查
```

| 部分 | 文件 | 解决什么问题 |
|---|---|---|
| [A] 夹爪位置 | `gripper/track.py` | SAM 2.1 base_plus 从少量人工锚点框出发，跟踪左右两只夹爪。每 50 帧一个窗口，用上一窗口掩码的框重新提示，经过锚点时用人工框纠正。每帧都有框，左右身份不会互换；夹爪被挡住时沿用上一帧的框并做标记 |
| [A] 夹爪开合 | `gripper/state.py` | YOLO11s-cls 对夹爪的方形裁剪图（在原 1280×720 帧上裁，256²）分类：open / closed / unknown |
| [A] 抓取事件 | `subtask_seg/pipeline/events.py` | 对 `1 − p_closed` 做中值滤波 + 滞回阈值，得到闭合区间。“开始张开的第一帧” = 从重新张开处往回找，第一帧高出闭合平台的位置，再减去分类器滞后 `ONSET_SHIFT` |
| [B] VLM 打分 | `subtask_seg/pipeline/vlm.py` | Qwen3-VL-8B 只当打分器，不生成文字。多选题读各选项字母的 log-prob（每个片段的字母顺序随机打乱，再减去每个选项在整段视频上的均值来去偏）；条件行问是非题；按件计数的行问“有几个”；空抓候选问“有没有拿起物体” |
| [C] 语法 DP | `subtask_seg/pipeline/tasksheet.py` `align.py` | 把 task sheet 展开成有序的 slot（固定次数 / 条件行 / 按件计数 / 最后的 Last frame 行），边界只能落在候选点上。代价 = −VLM log-prob + 候选类型代价 + 跳过 / 多用 slot 的代价 |
| [D] retry | `subtask_seg/pipeline/retry.py` | 空抓（闭合 ≥ 0.4 s、几乎没动、同一夹爪 3 s 内在同一位置再抓，且 VLM p(拿起) < 0.5），或放下后在原处重抓 → 该 slot 记为 suboptimal，recovery_t = 最后一次伸手的开始 |
| 导出 + 自查 | `export.py` `validate.py` | 输出首尾相接的 subtask 和 steps 区间；每次运行检查规范里的 5 项 + 格式，打印 `CHECK` |

### 3.1 怎么把边界定到 ±0.5 s 以内

大多数 subtask 以“夹爪开始张开的第一帧”结束。这是夹爪上的几何事件，所以直接从每只夹爪的开合曲线上读，不让时序模型去猜：

1. SAM 2 保证每一帧都有这只夹爪的框；分类器只看一张只含夹爪的小图，判断它开没开。
2. 分类器按单帧外观判断，相对“手指开始分开”会有一个固定的滞后。开合时间轴的标注规则就是“切换点 = 手指开始分开的第一帧”，
   所以 `subtask_seg/tools/measure_lag.py` 在时间轴单元上比较人工切换点和检测到的松手，量出这个滞后，作为 `--onset-shift`。
3. 两只夹爪都松开时，取 slot 内最晚的那次松开。不以松手结束的行（推、擦、压）用“夹爪从静止开始移动”的帧，再没有就退到 1 s 网格。

### 3.2 怎么对上 task sheet 的名字和次数

名字一律取 task sheet 原文，DP 只决定“是哪一行、从哪帧到哪帧”。夹爪事件知道“什么时候松手”，但不知道“在做哪一行”；VLM 正好相反，
它看得懂在做什么，但时间分辨率只有 1 s。所以**边界只允许落在夹爪事件上，行的身份由 VLM 分数决定**，两者在 DP 里合并。

次数的处理：
- 固定 count：直接展开。
- 按件计数：用 VLM 数数结果的众数展开，允许少用 / 多用（代价按数数概率）。
- 条件行：用还是不用，按是非题的概率。

### 3.3 怎么发现 retry

只看 [A] 的抓取事件：短闭合 + 夹爪几乎没动 + 同一夹爪 3 s 内在同一位置再抓。满足这些后，再由 VLM 做一次是非确认，避免把分类器抖动造成的短闭合当成空抓。
对于 task notes 写明“放下重抓算 retry”的行（比如餐具），放下后在原处再抓也算。叠毛巾、拉直毛巾时的松开再捏不算。

### 3.4 为什么这样选

规范里的边界几乎都锚定在夹爪开合上，所以把精度押在“夹爪在哪 + 开没开”上。task sheet 已经给了顺序和次数，
“分段 + 命名”就变成一个约束很强的对齐问题，可以用 DP 精确求解，而不是让模型自由生成分段。

---

## 4. 试过但没成功的方法

| 方法 | 结果 / 问题 |
|---|---|
| **Qwen3-VL 自由生成**（`Qwen_exp1.py`，输出在 `runs/exp1/`）：2 fps 抽帧、8 s 窗口（6 s 步长），给出完整 task sheet，让模型先写逐帧观察再输出 JSON 分段 | 时间分辨率最多 0.5 s，而且边界常常就是窗口的首尾。经常把整个窗口报成一个 subtask，例如 w01 把 300–675 帧整段报成 “Move one towel…”，`ends_when_seen: false`。夹爪开合 / 可见性的文字描述没法逐帧核对；窗口之间互相不一致，拼不成整段。→ 改成只读 log-prob 的打分器（[B]），边界交给夹爪事件 |
| **YOLOE 开放词表**（`yolo_exp1.py`）：用 “open / closed robot gripper” 两个文本提示 | 能找到夹爪，但分不出开合。现在只用来给训练帧预标夹爪位置（`gripper.prepare prelabel`） |
| **YOLO11 检测器直接检测 open / closed 夹爪**（v1，66 帧训练） | val mAP50 0.65；换到 12 号的场景（白色臂）后，约 70% 的帧没有框。→ 改成 SAM 2 跟踪 + 裁剪图分类（v2） |
| YOLO-World + ByteTrack 检测物体，用物体位移判断有没有拿起、关联到哪一行 | 26 号里几乎所有餐具都被叫成 teaspoon，类别对不上行。现在保留为 `--objects` 实验开关 |
| 只用抓取事件做 DP（没有 VLM） | 12 号的抓取被整团分进第一个 slot，±25 帧内 0%。松手帧本身是准的，错在“哪段属于哪行”。→ 因此加了 [B] |
| 用指尖关键点（YOLO-pose）测开口度 | 俯拍时指尖经常被挡，恰好在松手那一刻读不到 |

---

## 5. 怎么知道准不准

除了 12、26 号的手标参考，没有标准答案。所以先对各个模块分别量，再看端到端结果。

### 5.1 不需要参考的自查

- `subtask_seg/tests/synthetic_check.py`：用已知事件构造假的感知数据，断言边界误差 ≤ 3 帧、retry 被识别出来、条件行被跳过、次数和 steps 正确。目前通过。
- `validate.py`：每次运行都检查规范里的 5 项（例如“以松手结束的行，结束帧前后 0.5 s 内要有夹爪张开”），不满足就打印 `CHECK`。具体列在第 6 节。

### 5.2 [A] 夹爪跟踪：跟踪框 vs 留出的人工框（`python -m gripper.track report`，训练视频）

| 视频 | 人工框 | ok | extent（范围不同） | off（跟错地方） | 面积比中位数 | 中心偏移中位数 |
|---|---|---|---|---|---|---|
| 09_coil_and_stow_four_cables | 20 | 8 | 6 | 6 | 1.52 | 0.24 |
| 13_fold_three_pairs_of_pants | 8 | 3 | 4 | 1 | 0.98 | 0.26 |
| 14_full_wipe_down_sequence | 26 | 21 | 2 | 3 | 0.78 | 0.08 |
| 16_lace_and_tie_a_shoe | 8 | 4 | 2 | 2 | 0.85 | 0.22 |
| 21_portion_dry_goods_into_jars | 58 | 45 | 7 | 6 | 0.89 | 0.13 |
| **合计** | **120** | **81 (68%)** | **21 (18%)** | **18 (15%)** | | |

12、26 号只有锚点框（12 号 2 帧、26 号 4 帧），没有留出的框可以对比，跟踪质量只能看 `--vis` 叠加视频和告警（两次运行里 12 号各有 13 和 29 个，26 号各有 24 和 28 个）。

### 5.3 [A] 开合分类器

- 训练数据：24 个时间轴单元（来自 13、14、16、21 号，每个单元 500 帧，共 12000 帧逐帧标签）+ 164 个框，得到 train 2842 张 / val 860 张裁剪图，按片段划分。
- c1（yolo11s-cls，40 epoch，训练 2.2 分钟）：val top-1 **0.78**。c0（只用框训练，66 帧左右）是 0.68。val 里 unknown 只有 1 张，所以这个数基本就是 open / closed 二分类的准确率。

### 5.4 [A] 松手时刻（`tools/measure_lag.py`，人工切换点 vs 检测到的松手，结果在 `subtask_seg/lag.txt`）

| 集合 | 人工松手数 | 误差（帧） | median \|e\| | ±25 帧内 | 漏检 / 多检 |
|---|---|---|---|---|---|
| train 单元 | 17 | 大多在 −3…0 | 1 帧 | 100% | 0 / 2 |
| val 单元 | 3（来自 14、13 号） | −18, 0, −7 | 7 帧 | 100% | 3 / 4 |

由此得到 `ONSET_SHIFT = −7`。**注意 val 只有 3 个松手**，这个数不稳；train 上的 1 帧是在训练数据上量的，偏乐观。

### 5.5 端到端：和手标参考比较（`tools/eval_gt.py`）

参考答案是我在 task player 里逐帧标的：`data/12_fold_four_towels_GT.json`（9 段）和 `data/26_sort_cutlery_drawer_insert_GT.json`（10 段）。


| 视频 | 段数 GT / 预测 | 边界误差（预测 f_end − GT，帧） | median \|e\| | ±25 帧内 | status 一致 |
|---|---|---|---|---|---|
| 12_fold_four_towels | 9 / 8（漏 1 段） | +358, +94, −183, −62, +364, +61, +14 | 94 帧（1.9 s） | 1/7（14%） | 6/8 |
| 26_sort_cutlery_drawer_insert | 10 / 11（多 1 段） | −582, −302, +462, −353, −827, −352, −1070, +821, +538 | 538 帧（10.8 s） | 0/9 | 8/10 |
| 合计 | | n = 16 | 356 帧（7.1 s） | 1/16（6%） | 14/18 |

误差分布：|e| ≤ 25 帧的 1 个，25–100 帧 3 个，100–400 帧 6 个，> 400 帧 6 个。12 号以偏晚为主（平均 +92 帧），26 号以偏早为主（平均 −185 帧）。

**误差出在哪一环？** 把每个 GT 边界和离它最近的候选边界相比，相当于“如果 DP 把行都选对，最好能做到多少”：

| | 12 号 | 26 号 |
|---|---|---|
| 最近的松手候选 | median 12 帧，6/8 在 ±25 帧内 | median 87 帧，0/9 在 ±25 帧内（多数晚 40–190 帧） |
| 最近的松手 + 开始移动候选 | median 6 帧，8/8 | median 17 帧，6/9 |
| VLM 每秒片段的 top-1 行 = GT 行 | 16/48 = **33%** | 51/176 = **29%**（有 77 个片段被判成最后的 “Verify …” 行） |

（“开始移动”的候选很密，平均 1–2 s 就有一个，所以第二行有一部分是碰巧对上的，只能当作上限看。）

结论：
- 12 号的候选里，几乎每个真边界附近都有候选点，**错在 VLM 把行认错，导致 DP 选错了点**。
- 26 号除了这个问题，松手检测本身也偏晚，原因还没查清（可能是餐具放进插槽时夹爪被手臂 / 插槽挡住）。两个问题叠加在一起。

---

## 6. 已知错误（当前的 `subtask_seg/results/`，ONSET_SHIFT = −7）

**12_fold_four_towels**
- seq 1 “Move one towel from the left pile…”：预测结束在 10.08 s，GT 是 2.92 s（晚 7.2 s）；而且被误标为 suboptimal（recovery_t 4.48 s），GT 是 ok。
- seq 3 “Fold the towel lengthwise”：预测结束在 23.20 s，GT 是 26.86 s（早 3.7 s）；结束处前后 0.5 s 内没有夹爪张开（`CHECK [4]`）。
- seq 4 “Fold the strip widthwise”：GT 是 suboptimal（31.98 s 时一角翘起，没叠好），预测是 ok。这是“结果质量”的问题，夹爪规则看不出来。
- seq 5 “Pull the folded towel aside”：预测结束在 42.02 s，GT 是 34.74 s（晚 7.3 s）。
- **漏掉了** seq 6 “Drag the pile next to the folded towel”（GT 34.74–38.16 s）：三个条件行的 VLM 是非题都给出 p(yes) = 0.003，所以 DP 倾向于跳过条件行。
- 45.82 s、47.60 s 两个边界附近没有夹爪张开（`CHECK [4]`）。

**26_sort_cutlery_drawer_insert**
- **多出了**第二把 dinner spoon（3.94–34.0 s）：场景里只有一把 dinner spoon，DP 把后面的抓取凑成了第二把，导致后面所有行整体错位。
- 所有边界都差 6–21 s：fork #1 早 6.0 s（GT 47.7 s），fork #2 晚 9.2 s（GT 57.2 s），knife #2 早 16.5 s（GT 92.2 s），
  teaspoon #2 早 21.4 s（GT 118.3 s），wooden serving utensil 晚 16.4 s（GT 143.6 s），slide insert 晚 10.8 s（GT 155.2 s）。
- 预测的 knife #1 只有 66.42–67.14 s（0.72 s），这么短不可能是一次完整的放置。
- fork #2（recovery_t 52.6 s）和 wooden serving utensil（recovery_t 150.2 s）被误标为 suboptimal，GT 全部是 ok。
- 95.30 s 时夹爪 1 有一次空抓，落在一个标成 ok 的段里（`CHECK [5]`：是 retry 还是按住？）。
- 3.94 s 之前夹爪 1 被挡住，那个松手边界只是上界（`CHECK [occluded]`）。

**结构性问题**
- VLM 每 1 s 取一个 2 s、6 帧的片段，短于 1 s 的行只能靠相邻片段判断；相似的行（两种勺子、叉子 vs 刀）区分不开。
- 如果实际操作先做了后面的 step（顺序和 task sheet 不同），DP 不能打乱顺序，只会跳过或错配。
- 白色臂（YAM）上 SAM 2 的掩码常常连带相机支架 / 手腕，框偏大（09 号面积比 1.52）；夹爪被完全挡住时，框沿用上一帧。
- SAM 2 没有编译 CUDA 扩展（日志里有 `cannot import name '_C'`），跳过了掩码后处理（填洞），一般不影响框。

---

## 7. 成本

**硬件**：Brown CCV Oscar，NVIDIA GeForce RTX 3090（24 GB），`3090-gcondo` 分区。**API 费用为 0**（全部使用本地开源模型）。

| 步骤 | 12 号（49.7 s，2484 帧） | 26 号（177.7 s，8886 帧） |
|---|---|---|
| SAM 2 跟踪 L + R | 276–283 s（≈ 8.8 fps） | 974–1033 s（≈ 8.8 fps） |
| 开合分类 | 4968 张裁剪图 | 17772 张；两段合计 < 2 分钟 |
| [B]–[D] 分段（含 VLM，48 / 176 个片段） | 219 s | 273 s |
| **合计** | **≈ 9 分钟（≈ 11× 实时）** | **≈ 23 分钟（≈ 8× 实时）** |

一次性成本：c1 训练 2.2 分钟；`run_c1.sbatch` 全流程约 40 分钟。

**人工**（下表都是人工输入，不是对输出的修正；输出 JSON 的人工修正为 0）：

| 人工输入 | 数量 | 用时 |
|---|---|---|
| 训练片段的夹爪框 | 82 帧（21 号 35、14 号 18、09 号 13、13 号 8、16 号 8） | 2h+ |
| 开合时间轴 | 24 个单元 × 500 帧 = 12000 帧，只标切换点 | 1h |
| 必做视频的锚点框 | 12 号 2 帧、26 号 4 帧 | 20 min |
| 手标 GT 参考（只用于评估） | 12 号 9 段、26 号 10 段 | 2h |

---

## 8. 如果要标 1 万小时

按现在的速度（RTX 3090 上约 8–11 GPU 小时 / 视频小时），1 万小时 ≈ **8–11 万 GPU 小时**；VLM 每秒一次前向，总共约 3600 万次。瓶颈按优先级排：

1. **准确率要先于成本解决**：现在端到端只有 6% 的边界在 ±0.5 s 内，规模化之前先要把行识别做对。计划：
   - VLM 不再每秒独立打分，而是“给定候选边界，问这一整段在做哪一行”，让片段和语义单元对齐，帧数随段长增加；
   - 用物体级的证据区分相似的行（被放进去的是哪类餐具），而不是只看全局画面；
   - 条件行和数数的是非题改成多帧 + 自洽投票。现在像 p(yes) = 0.003 这样过度自信的答案会直接删掉整行。
2. **人工锚点框**：每段视频都要人画几帧。可以用已有的锚点 + 跟踪结果训练一个夹爪检测器，自动生成锚点，人只复核跟踪告警。
3. **SAM 2 逐帧跟踪是算力大头**（占总时间的 50–75%）：降到 25 fps 跟踪再插值，或者只在闭合区间附近逐帧跑；如果有机器人关节 / 夹爪指令数据，直接用它，跟踪和分类都可以省掉。
4. **开合分类器**：换新机器人 / 新相机时要补时间轴标注（一个单元几分钟）；有夹爪状态数据时可以直接生成标签。
5. **VLM**：换更小的模型或减少帧数，只在 DP 不确定（代价差很小）的区间去问。
6. **质量控制**：DP 代价和阈值现在是在 12、26 上手调的，需要一个独立的留出集来做网格搜索；按 DP 代价 + `CHECK` 数量，把最可疑的视频优先送人工复核（task player 可以直接载入预测 JSON 来修改）。
