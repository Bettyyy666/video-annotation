"""Qwen3-VL-8B (same model and loading as video-annotation/exp1) used as a SCORER, never as a generator:
one forward pass per question, read the probabilities of the answer tokens.

  * rows:     every CLIP_STEP_S seconds a CLIP_S-second clip (CLIP_FRAMES frames, VLM_WIDTH px wide) and a
              multiple-choice question "which task-sheet row is the robot doing?" over ALL rows of the
              task -> log p(row | clip). Option letters are shuffled per clip (seeded) so the model's letter
              preference averages out; align.py also removes each option's mean over the video.
  * yes/no:   p("Yes") vs p("No") - used for the conditional rows ("only if a pile is already at the
              back-right") on the first seconds, and for retry checks ("was an object lifted?").
  * counts:   p(digit) for "how many <object>s are on the table at the start" -> soft counts.

Everything is cached in results/vlm/<id>.json, so the DP can be re-run without a GPU.
"""
import io
import json
import os
import random
import re

import cv2
import numpy as np

from . import config as C

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class FrameCache:
    """Sequential decode once; keep every GRID-th frame resized to `width`, JPEG-compressed in memory."""

    GRID = 10  # frames (0.2 s at 50 fps)

    def __init__(self, video, width):
        self.width = width
        cap = cv2.VideoCapture(video)
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 50.0
        self.W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.scale = width / self.W
        self.jpg, f = {}, 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if f % self.GRID == 0:
                small = cv2.resize(img, (width, round(self.H * self.scale)), interpolation=cv2.INTER_AREA)
                self.jpg[f] = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            f += 1
        cap.release()
        self.T = f

    def get(self, f, box=None):
        """RGB PIL image of the cached frame nearest to f; optionally with a red box (original pixels)."""
        from PIL import Image
        g = int(round(min(max(f, 0), self.T - 1) / self.GRID) * self.GRID)
        g = min(self.jpg, key=lambda k: abs(k - g)) if g not in self.jpg else g
        img = cv2.imdecode(np.frombuffer(self.jpg[g], np.uint8), cv2.IMREAD_COLOR)
        if box is not None and np.all(np.isfinite(box)):
            x1, y1, x2, y2 = (int(round(v * self.scale)) for v in box[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))


class QwenScorer:
    def __init__(self, model_path=C.VLM_MODEL, load_in_4bit=False):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(model_path)
        kwargs = {"dtype": torch.bfloat16, "device_map": {"": 0}, "attn_implementation": "sdpa"}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True,
                                                               bnb_4bit_compute_dtype=torch.bfloat16)
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(model_path, **kwargs).eval()
        self.tok = self.processor.tokenizer
        self.name = os.path.basename(str(model_path).rstrip("/"))

    def token_id(self, s):
        ids = self.tok.encode(s, add_special_tokens=False)
        if len(ids) != 1:
            raise ValueError(f"answer '{s}' is not a single token: {ids}")
        return ids[0]

    def answer_logprobs(self, images, text, answers):
        """log p(answer) restricted to `answers` (single tokens), after the assistant turn starts."""
        torch = self.torch
        content = [{"type": "image"} for _ in images] + [{"type": "text", "text": text}]
        prompt = self.processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                    add_generation_prompt=True)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        ids = [self.token_id(a) for a in answers]
        return torch.log_softmax(logits[ids], dim=0).cpu().numpy()


def row_text(row):
    obj = row.obj or "-"
    return f"{row.name} [Step {row.step_n}: {row.step_name}; object: {obj}; ends when: {row.ends_when}]"


def clip_frames(t0_frame, fps):
    n = C.CLIP_FRAMES
    span = C.CLIP_S * fps
    return [int(round(t0_frame + span * k / (n - 1))) for k in range(n)]


def score_rows(scorer, cache, task_title, rows, seed=0):
    """-> list of clips {f0, f1, center, lp[len(rows)]} (lp = log p(row), original row order)."""
    fps, T = cache.fps, cache.T
    step = int(round(C.CLIP_STEP_S * fps))
    out = []
    starts = list(range(0, max(1, T - int(C.CLIP_S * fps)), step))
    for k, f0 in enumerate(starts):
        frames = [min(f, T - 1) for f in clip_frames(f0, fps)]
        perm = list(range(len(rows)))
        random.Random(seed * 100003 + k).shuffle(perm)
        opts = "\n".join(f"{LETTERS[i]}. {row_text(rows[p])}" for i, p in enumerate(perm))
        text = (f"The {len(frames)} images are frames {C.CLIP_S / (len(frames) - 1):.1f} s apart "
                f"(t = {frames[0] / fps:.1f}-{frames[-1] / fps:.1f} s) from a top-down camera over a table, "
                f"where a teleoperated two-arm robot does the task \"{task_title}\".\n"
                f"Which one of these task-sheet rows is the robot doing in these frames?\n{opts}\n"
                f"Answer with the letter only.")
        lp = scorer.answer_logprobs([cache.get(f) for f in frames], text, [LETTERS[i] for i in range(len(rows))])
        orig = np.empty(len(rows))
        orig[perm] = lp
        out.append({"f0": frames[0], "f1": frames[-1], "center": (frames[0] + frames[-1]) // 2,
                    "lp": [round(float(x), 4) for x in orig]})
        if k % 20 == 0:
            print(f"    vlm rows: clip {k + 1}/{len(starts)}", flush=True)
    return out


def yes_prob(scorer, images, question):
    lp = scorer.answer_logprobs(images, question + "\nAnswer Yes or No.", ["Yes", "No"])
    return float(np.exp(lp[0]))


def condition_of(row):
    m = re.search(r"only if (.+)$", str(row.count_text or ""), re.I)
    return m.group(1).strip().rstrip(".") if m else None


def start_frames(cache):
    return [cache.get(f) for f in clip_frames(0, cache.fps)]


def ask_condition(scorer, cache, cond):
    return yes_prob(scorer, start_frames(cache),
                    "These are the first frames of a top-down video of a robot workspace, before the robot "
                    f"starts. Is this true at the start: {cond}?")


def ask_count(scorer, cache, obj):
    lp = scorer.answer_logprobs(start_frames(cache),
                                "These are the first frames of a top-down video of a robot workspace. "
                                f"How many {obj}s can you see in total (on the table, not counting ones already "
                                "inside a container or holder)? Answer with a single digit.",
                                [str(d) for d in range(10)])
    return [round(float(np.exp(x)), 4) for x in lp]


def ask_lifted(scorer, cache, box_seq, c0, c1, side_word):
    """box_seq(f) -> gripper box at frame f (original pixels) for drawing."""
    fps = cache.fps
    a, b = max(0, c0 - int(0.4 * fps)), min(cache.T - 1, c1 + int(0.8 * fps))
    frames = [int(round(a + (b - a) * k / (C.CLIP_FRAMES - 1))) for k in range(C.CLIP_FRAMES)]
    imgs = [cache.get(f, box_seq(f)) for f in frames]
    return yes_prob(scorer, imgs,
                    f"These frames ({frames[0] / fps:.1f}-{frames[-1] / fps:.1f} s) show the robot gripper in the "
                    f"red box ({side_word} side of the image). It closes at t={c0 / fps:.1f} s and opens again at "
                    f"t={c1 / fps:.1f} s. While it was closed, did it hold an object and lift or move it "
                    "(not just close on nothing or touch it)?")


class VlmCache:
    """results/vlm/<id>.json; computes what is missing (lazy-loads the model only then)."""

    def __init__(self, path, video, model_path=C.VLM_MODEL, width=C.VLM_WIDTH, allow_compute=True):
        self.path, self.video, self.model_path, self.width = path, video, model_path, width
        self.allow = allow_compute
        self.d = json.load(open(path)) if os.path.exists(path) else {}
        self._scorer = self._cache = None

    def _need(self):
        if not self.allow:
            raise RuntimeError(f"VLM result missing in {self.path} and --no-vlm-compute was given")
        if self._scorer is None:
            print(f"  loading {self.model_path} ...", flush=True)
            self._scorer = QwenScorer(self.model_path)
            self._cache = FrameCache(self.video, self.width)
        return self._scorer, self._cache

    def save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(self.d, fh)
        os.replace(tmp, self.path)

    def rows(self, title, rows):
        names = [r.name for r in rows]
        if self.d.get("rows") != names or not self.d.get("clips"):
            s, c = self._need()
            self.d.update(rows=names, clips=score_rows(s, c, title, rows), width=self.width,
                          clip_s=C.CLIP_S, clip_step_s=C.CLIP_STEP_S, model=s.name)
            self.save()
        return self.d["clips"]

    def yes(self, key, fn, optional=False):
        y = self.d.setdefault("yes", {})
        if key not in y:
            if optional and not self.allow:
                return None
            s, c = self._need()
            y[key] = round(fn(s, c), 4)
            self.save()
        return y[key]

    def count(self, obj):
        cnt = self.d.setdefault("counts", {})
        if obj not in cnt:
            s, c = self._need()
            cnt[obj] = ask_count(s, c, obj)
            self.save()
        return cnt[obj]


def dump_images(images, path):
    """Debug helper: save the images a question saw."""
    os.makedirs(path, exist_ok=True)
    for k, im in enumerate(images):
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=85)
        open(os.path.join(path, f"{k}.jpg"), "wb").write(buf.getvalue())
