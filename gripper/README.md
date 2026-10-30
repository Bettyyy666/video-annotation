# gripper：夹爪定位 + 开合判断（模块 [A]）

**v2 结构（当前）**：SAM 2 跟踪左右两只夹爪 + YOLO11-cls 对夹爪裁剪图分类开合。

```
人工锚点框（少量） ──► [track.py] SAM 2 视频跟踪 ──► 每帧 L / R 夹爪框，左右身份不变
                                                                     │
                     夹爪裁剪图（原 1280×720 帧上裁，256×256） ◄──────┘
                                   │
          [state.py] YOLO11-cls：open / closed / unknown ──► 每帧 p_open(L), p_open(R) ──► sides.csv
```

为什么改：v1 的 YOLO 检测器每帧都要重新“找到”夹爪，没见过的场景就找不到（r1 在 12 号视频上
约 70% 的帧没框）。SAM 2 只要每只夹爪一个起始框就能一直跟下去，换场景也行，左右身份天然稳定；
开合判断变成对一张只含夹爪的小图做分类，比在整帧里同时找位置和判开合容易得多。

所有图像都来自原 MP4 解码出的 **1280×720 原分辨率帧**，按整数帧号寻址：SAM 2 的视频接口只读
JPEG 文件夹，所以每个窗口的帧写成质量 95 的 JPEG 给它；分类用的裁剪图直接从解码帧上裁。

## 数据与划分

- 训练视频：`common.py` 的 `TRAIN_VIDEOS`（09 线缆、14 擦桌、21 分装）里的随机片段（`prepare segments`）。
  可以加别的非必做视频（比如 `11_fold_five_t_shirts_from_a_pile.mp4`，白色臂 + 木桌，和 12 号同款）。
- 必做视频 `EVAL_VIDEOS`（12、26）：只画**锚点框**给跟踪器用，**任何东西都不会从它们导出去训练**
  （export_yolo、state export 都会跳过）。
- 验证集按片段划分（每段视频约 20% 的片段整段进 val）。

## 标注：两种

1. **框**（原来的标注工具 `http://localhost:8765/`）：给跟踪器当锚点。训练片段里每个片段最早的
   done 帧自动当锚点，其余 done 帧用来**检查跟踪质量**（IoU）。你已经标好的 66 帧直接可用。
2. **开合时间轴**（新页面 `http://localhost:8765/state`）：一个单元 = 一只夹爪（L 或 R）× 一个片段。
   左边是这只夹爪的裁剪图，右边是整帧（L 蓝框、R 橙框），下面是时间轴。逐帧/播放浏览，
   **只在状态改变的那一帧按 1/2/3**，之后的帧自动沿用，一个 10 s 片段（500 帧）几分钟就能标完，
   得到几百个逐帧标签。

### 时间轴规则

- open：两指之间能看到缝隙，且没夹东西；closed：两指并拢或夹住物体；unknown：手指被挡住/看不清。
- 闭合→张开：切换点放在**手指开始分开的第一帧**（= 规范里的“开始张开的第一帧”，不是完全张开；逐帧往回找）。
  张开→闭合：手指碰到一起或夹住物体的那一帧。
- 裁剪图里不是这只夹爪（跟踪框跑偏）：用 `B` 标成坏框区间，这些帧不进训练；跑偏处按 `A` 加锚点。
- 有模型预测时，时间轴第 1 行先显示模型猜的切换点（斜纹），检查/改完按 Enter 就变成你的标签。

### 时间轴快捷键

| 操作 | 键 |
|---|---|
| 在当前帧设切换点 | `1` open　`2` closed　`3` unknown |
| 删掉决定当前状态的切换点 | `Del` / `Backspace` |
| 前后 1 帧 / 10 帧，首尾 | `←` `→`，加 `Shift`；`Home` `End` |
| 播放 / 暂停，调速 | `Space`，`+` `-` |
| 点击/拖动时间轴跳转，滚轮缩放 | 鼠标 |
| 上/下一个切换点，上/下一个跟踪告警 | `[` `]`，`{` `}` |
| 坏框区间：起点、终点 / 删除光标处区间 | `B` … `B` / `Shift+B` |
| 在当前帧加锚点（打开框工具改框，Enter 后重跑 track） | `A` |
| 单元完成 → 下一个未完成单元 / 上下单元 | `Enter` / `N` `P` |

## 流程（集群上，`video-annotation/` 目录下；conda 环境 `annotation`）

```bash
export YOLO=/users/rzhou52/data/users/rzhou52/YOLO11
# SAM 2 权重默认读 checkpoints/sam2.1_hiera_<size>.pt；在别处就加 --ckpt <path>

# ---- 训练片段
# 1. 跟踪所有已有 done 帧的片段（默认 base_plus；--sam2 small 更快）
python -m gripper.track segments
#    输出每个片段 L/R 跟踪到的帧数、告警数，以及和其余人工框的 IoU（中位数应 > 0.7）
python -m gripper.track report       # IoU 低时：区分“框的范围不同”(extent) 和“跟错地方”(off)，并出图

# 2. 第 0 轮分类器：只用已标的框（66 帧左右）训一个粗分类器，再预测所有跟踪帧 → 时间轴预标注
python -m gripper.state export
python -m gripper.state train --name c0 --epochs 30
python -m gripper.state predict --model runs/gripper/cls/c0/weights/best.pt

# 3. 打开 http://localhost:8765/state 标时间轴（label_tool 照常启动 + SSH 隧道）
python -m gripper.label_tool --port 8765

# 4. 训正式分类器（yolo11s-cls；--weights $YOLO/yolo11n-cls.pt 更小）
python -m gripper.state export
python -m gripper.state train --name c1
python -m gripper.state predict --model runs/gripper/cls/c1/weights/best.pt
#    再看时间轴里模型和你的标签不一致的地方，修正 → 重复 4

# 想加 11 号视频（白色臂、木桌）：
python -m gripper.prepare segments --videos 11_fold_five_t_shirts_from_a_pile.mp4 --n-seg 4
#    在框工具里每个新片段至少标 1 帧（两只夹爪的框）→ track segments → 时间轴

# ---- 必做视频（12、26）
python -m gripper.prepare anchor --video 12_fold_four_towels.mp4 --frames 0
#    框工具里给这一帧画好两只夹爪的框（状态随便选，不会用于训练），Enter
python -m gripper.track video --videos 12_fold_four_towels.mp4
python -m gripper.state predict --model runs/gripper/cls/c1/weights/best.pt --videos 12_fold_four_towels.mp4 --vis
#    看 runs/gripper/pred/12_fold_four_towels.vis.mp4 和时间轴里的告警（红色刻度）；
#    跟丢/跑偏处按 A 加锚点、改框，然后重跑 track video 和 predict
```

## 跟踪的细节（track.py）

- 片段/视频切成 50 帧的窗口；每个窗口用**框**提示 SAM 2：锚点帧用人工框，其余用上一窗口最后
  一帧掩码的框（框提示让掩码停在夹爪头上，不会顺着手臂越长越大）。
- 从第一个锚点向后跑到结尾，再向前反向跑到开头；路过后面的锚点时用人工框重新提示（=纠错）。
- **每一帧都有框**：有掩码就用掩码的框，不限大小（夹爪抬到镜头前会大好几倍）；掩码消失（夹爪
  被完全挡住/出画）时沿用上一帧的框，标为 filled；相邻两帧框面积突变 2.5 倍以上或 L、R 两框重叠
  （IoU>0.5）时保留框但标为 suspect。filled / suspect 段的起点写进告警（时间轴上的红色刻度，
  `{` `}` 跳转），这些帧不进分类器训练集，需要时在那里按 `A` 补锚点。
- 输出 `runs/gripper/tracks/<视频>.json`：每帧 `L` / `R` 的 `[x1, y1, x2, y2, 掩码平均概率, 标记]`（0 正常，1 suspect，2 filled）。

## 输出

- `runs/gripper/items.json`、`segments.json`、`frames/`：框标注（同 v1）
- `runs/gripper/tracks/<视频>.json`：SAM 2 跟踪结果；`crops/`：时间轴页面用的裁剪图缓存
- `runs/gripper/states.json`：时间轴标签（切换点 + 坏框区间 + 是否完成）
- `runs/gripper/dataset_cls/`、`cls/<name>/weights/best.pt`：分类数据集与模型
- `runs/gripper/pred/<视频>.json`（每帧每侧 `[p_open, p_closed, p_unknown]`）、`.sides.csv`
  （列同 v1：state、conf、p_open、框；`p_open` 空 = 缺失或 unknown，留给下游插值/HMM）、`.p_open.png`、`--vis` 叠加视频

`p_open = p_open_cls / (p_open_cls + p_closed_cls)`；最大类是 unknown 时记为缺失。

## v1（保留，可选）

`prepare mine` / `export_yolo` / `train` / `infer` 是 v1 的 YOLO 检测器路线，代码保留；
`models.py` 的 YOLOE 预标注仍可用来给新片段画第一帧的框。

## 已知限制

- 锚点需要人工：每个片段至少 1 帧框，必做视频至少第 0 帧；跟踪跑偏处要补锚点。README 里记为人工输入。
- 单帧分类器分不出“正在张开”和“正在闭合”（半开的样子一样），只判当前外观；事件时间的固定滞后
  之后用 Δ 校正。
- SAM 2 的掩码可能连到夹住的物体（拎起的毛巾）上，框变大 → 被丢弃并告警；这种帧需要补锚点。
