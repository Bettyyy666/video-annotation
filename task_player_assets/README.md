# 精确帧播放器

源码只维护本目录这一版：`player.html`、`app.js`、`core.js` 和 `serve.py`。`task_player.py` 负责准备视频数据，不再复制页面、服务脚本或启动文件到 `players/`。

双击 `task_player_assets/打开播放器.command`（macOS）。它会启动仅本机访问的轻量服务，并在浏览器打开 `http://127.0.0.1:8766/index.html`。请保持启动窗口打开；Ctrl+C 停止。不需要 GPU 或模型。播放器源码统一在 `task_player_assets/`，`players/` 仅保存视频帧、视频信息和解码配置。

修复加载问题后，推荐使用这个本机网页地址，不再通过静态 HTML 预览或直接打开文件来启动。重复双击启动文件会重新打开已运行的同一个播放器，不会重复占用端口。若已有旧页面，请打开上述新地址。

页面现在会明确报告初始化错误；即使脚本未执行，也会显示启动说明，不再只留下“载入帧”。旧 file 页面与本机网页属于不同的浏览器存储来源，已有草稿可从旧页面下载后再载入新页面。

导入下一个视频：展开页面顶部“导入 / 切换视频”，选择视频文件和对应任务，点击“导入视频”。完成后点击“打开所选视频”；也可用列表切回原视频。支持原始 50 FPS 视频、最大 2 GB，处理仅发生在本机。导入后的帧和视频信息存入 `players/12_fold_four_towels/imports/`。按视频内容指纹分别保存浏览器草稿，切换时会先保存当前编辑。

只显示一张大图：

| 操作 | 功能 |
| --- | --- |
| → / ← | 前后跳 10 帧，50 FPS 下是 0.2 秒 |
| Shift + → / ← | 前后跳 1 帧 |
| Space | 自动每次前进 10 帧 / 暂停 |
| Home / End | 第一帧 / 最后一帧 |
| 输入帧号后 Enter | 直接跳转 |
| B | 将当前帧记录为所选 subtask 的结束边界 |
| R | 将当前帧记为正在编辑的 subtask 的 recovery_t |

在文本输入框或 task 下拉框中时，方向键保留编辑功能。点击大图即可恢复播放器快捷键。

播放器使用逐帧顺序解码生成的 JPEG（默认保留原分辨率、quality=95），每张图对应原视频的一个帧号，不依赖浏览器对视频的近似 seek。自动前进会等待当前图片完成加载，磁盘慢时可能比视频时钟慢，但不会把另一帧显示成当前帧号。

下方“记录 task ground truth”默认折叠。展开后选择 task，停在结束帧并按 B。第一段从 0 开始，后续段接上一段结束帧；自动计算 `f_start`、`t_start`、`t_end`，并由连续 subtask 汇总 `steps`。不会自动补可选步骤或次数。

每段的“编辑全部字段”可修改 JSON 中全部 subtask 字段，包括名称、step、occurrence、object、status、note 和 recovery_t。时间和帧号联动；修改开始边界会同步上一段结束，修改结束边界会同步下一段开始。修改后点击“保存 subtask 修改”。重复进入相同步骤时，可勾选“开始新的 step 区间”。

`recovery_t` 以秒填写，也可以输入 recovery 帧号，或在播放器停到恢复帧后按 R。设置 recovery 会将状态设为 suboptimal；恢复点必须位于当前编辑的 subtask 内。页面实时展示汇总后的 steps 和完整 JSON。

可以点击“载入你提供的 JSON”加载本次提供的九段数据，也能从文件或文本框导入 `{episode, steps, subtasks}`。导入时重新计算时间与 steps；替换已有标注前会提示，并保留一份浏览器内备份。模板的结束帧 2475 会原样保留，页面提醒它与视频末帧 2483 的差异，需人工核对。

“下载当前 JSON（允许未完成）”随时输出上述格式，保留 `robot`、`outcome`、`note` 等空值为 null。“导出标注 JSON”执行完整覆盖、outcome、任务字段等校验。另可下载 `.ground_truth.draft.json` 保留人工耗时等工作信息，稍后载入继续。已保存的修改会自动存入浏览器，兼容旧版草稿；仍建议下载备份。

重新生成（从 `video-annotation/` 目录）：

```bash
python3 -m venv .venv-player
.venv-player/bin/python -m pip install opencv-python-headless
.venv-player/bin/python task_player.py \
  --video ../data/12_fold_four_towels.mp4 \
  --task-sheet ../task_sheets.json
```

相同视频和参数重复运行会复用已有帧，只更新视频信息。页面与脚本在请求时从 `task_player_assets/` 读取，修改 HTML / JS 后刷新浏览器即可；修改 `serve.py` 后需要重启服务。换视频、分辨率或 JPEG 质量时使用新的 `--out` 目录。`--width 1280` 可减小图片体积，默认不缩放。

验证：

```bash
node --test tests/task_player_core.test.js
.venv-player/bin/python -m unittest discover -s tests -p 'test_task_player*.py' -v
```

测试涵盖 10 帧/1 帧跳转、首尾限制、完整覆盖、共享边界、retry、草稿来源校验、JSON 导入、空值保留、边界编辑联动、steps 汇总，以及真实解码一段合成视频后对帧数、末帧及像素顺序的检查。另在浏览器中用合成视频实测初始帧加载、键盘跳帧、九段模板导入、时间修改与 recovery 编辑保存。未让 agent 查看任务视频或填写 ground truth。
