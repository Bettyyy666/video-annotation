"""Experiment 1: let Qwen3-VL describe short windows of a video, then match to the task sheet.

- Samples ~2 fps, keeping the ORIGINAL frame number and time of every sampled frame.
- Splits the samples into short overlapping windows (not the whole video at once).
- Gives the model the COMPLETE task sheet entry (all subtasks, ends_when, count, notes).
- Asks for observations first, then matching; "unknown" is allowed.
- Saves the raw model output for manual checking. No post-processing.

Usage (on the GPU machine):
    python exp1_observe.py                      # defaults below
    python exp1_observe.py --dry-run            # only sample frames + write prompts, no model
"""
import argparse
import json
import time
from pathlib import Path

import cv2
from PIL import Image

DATA = "/users/rzhou52/data/users/rzhou52/video-annotation/data"
MODEL = "/users/rzhou52/data/users/rzhou52/Qwen3-VL-8B-Instruct"

PROMPT = """You are looking at a robot manipulation video. A bimanual teleoperated robot is filmed from above (top-down view of a table).
You will see {n} frames sampled every {interval}s from the window t = {t0:.3f}s to {t1:.3f}s of the video.
Each image is preceded by its original frame number and timestamp. This window may start or end in the middle of an action.

Below is the COMPLETE task sheet for this task. It is only a REFERENCE for what may happen.
It does NOT mean every action happens in this window, in this order, or the expected number of times.

TASK SHEET:
{sheet}

Answer in two parts.

PART 1 - OBSERVATIONS (from the images only, do not use task sheet names here).
One line per frame, or per run of frames where nothing changes:
  frame <f> (t=<t>s): left gripper: <open/closed/holding what/not visible/unclear>; right gripper: ...; towel and other objects: <position and shape>
If you cannot see something clearly, write "unclear". Do not guess.

PART 2 - MATCHING.
Group your observations into actions and match each one to a task sheet subtask.
Output a JSON list inside ```json ... ```. Each item:
{{"start_frame": <int>, "end_frame": <int>, "observed_action": "<what you saw>",
  "subtask": "<exact subtask text from the task sheet, or \\"unknown\\">",
  "ends_when_seen": <true / false / "unclear">,
  "evidence_frames": [<int>, ...], "uncertain": "<what is unclear, or null>"}}

Rules:
- Report only actions you actually see in these frames.
- Do NOT add actions because the task sheet expects them. Do NOT fill up the expected counts.
- If you cannot tell which subtask it is, use "unknown".
- Use only frame numbers shown above.
"""


def load_task(path, video):
    """Video "12_fold_four_towels[_15s].mp4" -> tasks[11]; checks the name after the number matches."""
    num, _, rest = Path(video).stem.partition("_")
    t = dict(json.load(open(path))["tasks"][int(num) - 1])
    if not rest.startswith(t["task"]):
        raise ValueError(f"video {Path(video).name} does not match task #{num} '{t['task']}'")
    t.pop("data_folders", None)  # not useful to the model
    return t


def init_results(task, video):
    """Empty results skeleton in the annotation format: every step, and every subtask expanded by its count.

    Fixed counts ("4", "4, one per box") give that many occurrences. Variable counts ("per item", "0–1, only if ...")
    give one placeholder whose note holds the count text, to be duplicated or deleted after watching the video.
    """
    steps = [{"step_n": s["n"], "step_name": s["title"], "t_start": 0.0, "t_end": 0.0, "f_start": 0, "f_end": 0}
             for s in task["steps"]]
    subtasks = []
    for st in task["subtasks"]:
        num = st["count"].split(",")[0].strip()
        n, note = (int(num), None) if num.isdigit() else (1, f"count: {st['count']}")
        for k in range(1, n + 1):
            subtasks.append({"seq": len(subtasks) + 1, "step_n": st["step"], "name": st["subtask"],
                             "occurrence": k, "object": st["object"], "object_detail": None,
                             "t_start": 0.0, "t_end": 0.0, "f_start": 0, "f_end": 0,
                             "status": "ok", "recovery_t": None, "note": note})
    return {"episode": {"episode_id": Path(video).stem, "robot": None, "outcome": None},
            "steps": steps, "subtasks": subtasks}


def sample_frames(video, interval, width):
    """Read the video sequentially; keep every step-th frame (step = interval * fps, an integer),
    with its original frame number and time. The last decoded frame (end of video) is always kept."""
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    step = round(interval * fps)
    if abs(step - interval * fps) > 1e-6:
        print(f"WARNING: interval {interval}s = {interval * fps} frames at {fps} fps; using step {step}")

    def to_sample(bgr, f):
        img = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        img = img.resize((width, round(img.height * width / img.width)))
        return {"frame": f, "t": round(f / fps, 3), "image": img}

    samples, f, last = [], 0, None
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if f % step == 0:
            samples.append(to_sample(bgr, f))
        last = bgr
        f += 1
    cap.release()
    if last is not None and samples[-1]["frame"] != f - 1:
        samples.append(to_sample(last, f - 1))
    return samples, fps, f  # f = number of decoded frames


def make_windows(samples, window_sec, stride_sec):
    last_t = samples[-1]["t"]
    windows, t0 = [], 0.0
    while True:
        w = [s for s in samples if t0 <= s["t"] < t0 + window_sec]
        windows.append(w)
        if t0 + window_sec > last_t:
            break
        t0 += stride_sec
    return windows


def build_messages(window, sheet_text, interval):
    text = PROMPT.format(n=len(window), interval=interval, t0=window[0]["t"], t1=window[-1]["t"], sheet=sheet_text)
    content = [{"type": "text", "text": text}]
    for s in window:
        content.append({"type": "text", "text": f"\nFrame {s['frame']} (t={s['t']:.3f}s):"})
        content.append({"type": "image"})
    return [{"role": "user", "content": content}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=f"{DATA}/26_sort_cutlery_drawer_insert.mp4")
    ap.add_argument("--task-sheet", default=f"{DATA}/task_sheets.json")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--out", default="runs/exp1")
    ap.add_argument("--interval", type=float, default=0.2, help="seconds between sampled frames")
    ap.add_argument("--window-sec", type=float, default=8.0)
    ap.add_argument("--stride-sec", type=float, default=6.0)
    ap.add_argument("--image-width", type=int, default=640)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--dry-run", action="store_true", help="no model; only frames and prompts")
    args = ap.parse_args()

    out = Path(args.out)
    frames_dir = out / Path(args.video).stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    task = load_task(args.task_sheet, args.video)
    sheet_text = json.dumps(task, indent=1, ensure_ascii=False)
    samples, fps, n_frames = sample_frames(args.video, args.interval, args.image_width)
    windows = make_windows(samples, args.window_sec, args.stride_sec)
    print(f"fps={fps} decoded_frames={n_frames} samples={len(samples)} windows={len(windows)}")
    
    # Save sampled frames so the outputs can be checked against the video.
    for s in samples:
        s["image"].save(frames_dir / f"f{s['frame']:06d}_t{s['t']:07.3f}.jpg", quality=90)
    
    
    # initialize results json
    results = init_results(task, args.video)
    json.dump(results, open(out / f"{Path(args.video).stem}.json", "w"), indent=1, ensure_ascii=False)
    return 

    if not args.dry_run:
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        processor = AutoProcessor.from_pretrained(args.model)
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map="auto"
        )

    meta = {"video": str(args.video), "fps": fps, "decoded_frames": n_frames, "task": task["task"],
            "model": args.model, "args": vars(args), "windows": []}

    for i, w in enumerate(windows):
        messages = build_messages(w, sheet_text, args.interval)
        prompt = (messages[0]["content"][0]["text"] if args.dry_run else
                  processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        (out / f"w{i:02d}_prompt.txt").write_text(prompt)

        raw, secs = "", 0.0
        if not args.dry_run:
            inputs = processor(text=[prompt], images=[s["image"] for s in w], return_tensors="pt").to(model.device)
            t_start = time.time()
            with torch.no_grad():
                ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
            secs = time.time() - t_start
            raw = processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            (out / f"w{i:02d}_raw.txt").write_text(raw)
            print(f"window {i}: frames {w[0]['frame']}-{w[-1]['frame']} "
                  f"input_tokens={inputs['input_ids'].shape[1]} gen_sec={secs:.1f}")

        meta["windows"].append({"window": i, "frames": [s["frame"] for s in w],
                                "times": [s["t"] for s in w], "gen_sec": round(secs, 1), "raw_output": raw})

    json.dump(meta, open(out / "run.json", "w"), indent=1, ensure_ascii=False)
    print(f"saved to {out}")


if __name__ == "__main__":
    main()