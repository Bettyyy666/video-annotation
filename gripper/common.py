"""Shared bits for the gripper open/closed detector: paths, classes, frame reading, label store.

Everything works on ORIGINAL full-resolution frames (1280x720) decoded straight from the MP4,
addressed by integer frame index. No resized intermediate frames are ever used.
"""
import json
import os
import threading
from pathlib import Path

import cv2

# ---- cluster defaults (override with CLI flags) -------------------------------------------
ROOT = "/users/rzhou52/data/users/rzhou52"
DATA = f"{ROOT}/video-annotation/data"
YOLO_DIR = f"{ROOT}/YOLO11"
PROJECT = f"{ROOT}/video-annotation/runs/gripper"

# Training classes (what the detector learns).
CLASSES = ["gripper_open", "gripper_closed", "gripper_unknown"]
CLS_ID = {c: i for i, c in enumerate(CLASSES)}
# Extra label used only while annotating: "a gripper is here, state not decided yet".
# YOLOE pre-labels come in as PENDING; every PENDING box must be turned into one of CLASSES
# (or deleted) before a frame can be marked done. PENDING is never exported for training.
PENDING = "gripper"
LABELS = CLASSES + [PENDING]

# Videos the detector is trained on (random segments of them). The required videos 12 / 26 are
# deliberately not here, so that later boundary errors measured on them are not optimistic.
TRAIN_VIDEOS = [
    "09_coil_and_stow_four_cables.mp4",
    "14_full_wipe_down_sequence.mp4",
    "21_portion_dry_goods_into_jars.mp4",
]
# Required (evaluation) videos: only anchor boxes for tracking are drawn here; nothing from them is
# ever exported for training.
EVAL_VIDEOS = ["12_fold_four_towels.mp4", "26_sort_cutlery_drawer_insert.mp4"]


def is_eval(video):
    stem = Path(video).stem
    return any(stem.startswith(Path(v).stem) for v in EVAL_VIDEOS)  # also *_15s clips


# ---- SAM 2 ----------------------------------------------------------------------------------
SAM2_DIR = str(Path(__file__).resolve().parent.parent / "checkpoints")  # video-annotation/checkpoints
SAM2_MODELS = {  # name: (checkpoint, config name inside the sam2 package)
    "tiny": (f"{SAM2_DIR}/sam2.1_hiera_tiny.pt", "configs/sam2.1/sam2.1_hiera_t.yaml"),
    "small": (f"{SAM2_DIR}/sam2.1_hiera_small.pt", "configs/sam2.1/sam2.1_hiera_s.yaml"),
    "base_plus": (f"{SAM2_DIR}/sam2.1_hiera_base_plus.pt", "configs/sam2.1/sam2.1_hiera_b+.yaml"),
    "large": (f"{SAM2_DIR}/sam2.1_hiera_large.pt", "configs/sam2.1/sam2.1_hiera_l.yaml"),
}
SIDES = ("L", "R")

# ---- gripper crops for the open/closed classifier ------------------------------------------
CROP_PAD = 0.25   # square crop side = max(box w, h) * (1 + CROP_PAD), centred on the box
CROP_SIZE = 256   # crops are cut from the full-res frame, then resized to this (classifier input 224)


def crop(img, box, size=CROP_SIZE, pad=CROP_PAD):
    """Square crop around a gripper box from a full-resolution frame (edges padded by replication)."""
    import numpy as np
    x1, y1, x2, y2 = box[:4]
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1, 16) * (1 + pad) / 2
    a, b = int(round(cx - half)), int(round(cy - half))
    c, d = int(round(cx + half)), int(round(cy + half))
    H, W = img.shape[:2]
    padw = max(0, -a, -b, c - W, d - H)
    if padw:
        img = cv2.copyMakeBorder(img, padw, padw, padw, padw, cv2.BORDER_REPLICATE)
        a, b, c, d = a + padw, b + padw, c + padw, d + padw
    out = img[b:d, a:c]
    if size:
        out = cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA if out.shape[0] > size else cv2.INTER_LINEAR)
    return np.ascontiguousarray(out)


# ---- frame reading -----------------------------------------------------------------------
class FrameReader:
    """Frame-exact random access into one video, at original resolution.

    Reads forward when the target is a bit ahead of the current position (cheap and always exact),
    otherwise seeks with CAP_PROP_POS_FRAMES. `verify_seek` checks seeking against sequential decode
    on this machine's OpenCV build; run it once (`python -m gripper.common verify <video>`).
    """

    FORWARD_LIMIT = 150  # read forward instead of seeking if target is within this many frames

    def __init__(self, path):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise IOError(f"cannot open {self.path}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.n_meta = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))  # may be off by one vs decodable
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.pos = 0  # index of the frame the next read() returns
        self.lock = threading.Lock()

    def get(self, f):
        """BGR uint8 array (h, w, 3) of frame f, or None if it cannot be decoded."""
        with self.lock:
            if not (self.pos <= f < self.pos + self.FORWARD_LIMIT):
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, f)
                self.pos = f
            img = None
            while self.pos <= f:
                ok, img = self.cap.read()
                if not ok:
                    self.pos = 10 ** 9  # force a seek next time
                    return None
                self.pos += 1
            return img

    def __iter__(self):
        """Sequential decode of the whole video: yields (frame_index, image)."""
        cap = cv2.VideoCapture(self.path)
        f = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            yield f, img
            f += 1
        cap.release()


def verify_seek(path, n=30, seed=0):
    import hashlib
    import random
    seq = [hashlib.md5(img.tobytes()).hexdigest() for _, img in FrameReader(path)]
    r = FrameReader(path)
    idx = random.Random(seed).sample(range(len(seq)), min(n, len(seq))) + [0, len(seq) - 1]
    bad = [i for i in idx if (im := r.get(i)) is None or hashlib.md5(im.tobytes()).hexdigest() != seq[i]]
    r2 = FrameReader(path)  # also exercise pure seeks (backwards order)
    bad += [i for i in sorted(idx, reverse=True)
            if (im := r2.get(i)) is None or hashlib.md5(im.tobytes()).hexdigest() != seq[i]]
    print(f"{path}: decodable={len(seq)} meta={r.n_meta} size={r.w}x{r.h} fps={r.fps} "
          f"seek mismatches={sorted(set(bad))}")
    return not bad


class Readers:
    """Cache of FrameReader per video name (file name inside the data dir)."""

    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self._r = {}

    def __call__(self, video):
        if video not in self._r:
            # training videos live in data/train/, the required videos in data/
            p = self.data_dir / video
            if not p.exists() and (self.data_dir / "train" / video).exists():
                p = self.data_dir / "train" / video
            self._r[video] = FrameReader(p)
        return self._r[video]


# ---- label store -------------------------------------------------------------------------
# project/
#   items.json            list of items (source of truth for labels)
#   frames/<stem>/f000123.png   lossless full-res frames used for labeling + training
#
# item = {"id": "<stem>/f000123", "video": "x.mp4", "frame": 123, "t": 2.46,
#         "status": "todo" | "done" | "skip",
#         "seg": 2,                # index into segments.json[video] (None = outside any segment)
#         "source": "segment" | "mined:<reason>" | "manual",
#         "boxes": [{"cls": "gripper_open", "xyxy": [x1, y1, x2, y2], "by": "human" | "<model>", "conf": 0.8}]}
#   segments.json         {"<video>": [[start_frame, end_frame], ...]}  (end exclusive)


def item_id(video, frame):
    return f"{Path(video).stem}/f{frame:06d}"


class Store:
    def __init__(self, project):
        self.dir = Path(project)
        self.path = self.dir / "items.json"
        self.lock = threading.Lock()
        self.items = json.loads(self.path.read_text()) if self.path.exists() else []
        self.index = {it["id"]: k for k, it in enumerate(self.items)}
        self.seg_path = self.dir / "segments.json"
        self.segments = json.loads(self.seg_path.read_text()) if self.seg_path.exists() else {}

    def seg_of(self, video, frame):
        for k, (s, e) in enumerate(self.segments.get(video, [])):
            if s <= frame < e:
                return k
        return None

    def frame_path(self, it):
        return self.dir / "frames" / f"{it['id']}.png"

    def add(self, video, frame, fps, source, boxes=None, seg=-1):
        iid = item_id(video, frame)
        if iid in self.index:
            return None
        it = {"id": iid, "video": video, "frame": int(frame), "t": round(frame / fps, 3),
              "seg": self.seg_of(video, frame) if seg == -1 else seg,
              "status": "todo", "source": source, "boxes": boxes or []}
        self.index[iid] = len(self.items)
        self.items.append(it)
        return it

    def save(self):
        with self.lock:
            self.dir.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.items, indent=1))
            os.replace(tmp, self.path)  # atomic: a crash never leaves a half-written file
            tmp = self.seg_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.segments, indent=1))
            os.replace(tmp, self.seg_path)

    def ensure_frames(self, readers, items=None):
        """Write the lossless PNG for every item that does not have one yet (sequential per video)."""
        todo = [it for it in (items or self.items) if not self.frame_path(it).exists()]
        for it in sorted(todo, key=lambda x: (x["video"], x["frame"])):
            img = readers(it["video"]).get(it["frame"])
            if img is None:
                print(f"WARNING: cannot decode {it['id']}, dropping it")
                it["status"] = "skip"
                continue
            p = self.frame_path(it)
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), img, [cv2.IMWRITE_PNG_COMPRESSION, 1])
        return len(todo)


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "verify":
        ok = all([verify_seek(p) for p in sys.argv[2:]])
        sys.exit(0 if ok else 1)
    print("usage: python -m gripper.common verify <video.mp4> [...]")
