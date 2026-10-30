"""Detector wrappers. Both return, per image, a list of *gripper instances*:

    {"xyxy": [x1, y1, x2, y2], "cls": <best class>, "conf": <best score>,
     "scores": {"gripper_open": s, "gripper_closed": s, "gripper_unknown": s}}

Raw detections are produced with per-class NMS, so the same gripper can carry an "open" box and a
"closed" box. `group_instances` merges boxes that overlap (IoU > 0.5) into one instance and keeps the
best score of every class, which later gives p_open = s_open / (s_open + s_closed).

Two kinds of model:
  * yoloe:<weights.pt>   open-vocabulary YOLOE-11 with the text prompt "robot gripper" -- only for
                         round-0 pre-labels. Its boxes are imprecise and its open/closed guesses are
                         unreliable, so it outputs the PENDING class "gripper" (location only); the
                         annotator fixes the box and sets the state in the label tool.
  * <weights.pt>         a YOLO11 detector trained on our 3 classes (round >= 1 and final use)
"""
import os
from contextlib import contextmanager
from pathlib import Path

from .common import CLASSES, LABELS, PENDING, YOLO_DIR

# YOLOE text prompt -> label. Location only; the state is decided by the annotator.
YOLOE_PROMPTS = {"robot gripper": PENDING}


@contextmanager
def _cwd(d):
    old = os.getcwd()
    os.chdir(d)
    try:
        yield
    finally:
        os.chdir(old)


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def group_instances(dets, iou_thr=0.5, margin=0.0):
    """dets: list of (cls_name, conf, xyxy). Greedy grouping by IoU, highest confidence first.

    If the best open/closed scores of a group are within `margin`, the instance is called unknown.
    """
    dets = sorted(dets, key=lambda d: -d[1])
    used = [False] * len(dets)
    out = []
    for i, (c, s, b) in enumerate(dets):
        if used[i]:
            continue
        scores = {k: 0.0 for k in LABELS}
        for j in range(i, len(dets)):
            if not used[j] and iou(b, dets[j][2]) > iou_thr:
                used[j] = True
                scores[dets[j][0]] = max(scores[dets[j][0]], dets[j][1])
        cls = c
        if c in ("gripper_open", "gripper_closed") and abs(scores["gripper_open"] - scores["gripper_closed"]) < margin:
            cls = "gripper_unknown"
        out.append({"xyxy": [round(float(v), 1) for v in b], "cls": cls, "conf": round(float(s), 4),
                    "scores": {k: round(float(v), 4) for k, v in scores.items()}})
    return out


class Detector:
    def __init__(self, spec, device=None, imgsz=1280, conf=0.05, max_inst=4, margin=0.0):
        """spec: 'yoloe:/path/yoloe-11l-seg.pt' or '/path/best.pt'."""
        import torch
        self.device = device if device is not None else (0 if torch.cuda.is_available() else "cpu")
        self.imgsz, self.conf, self.max_inst, self.margin = imgsz, conf, max_inst, margin
        self.half = self.device != "cpu"
        if spec.startswith("yoloe:"):
            from ultralytics import YOLOE
            w = Path(spec[len("yoloe:"):])
            if not w.is_file():
                raise SystemExit(f"YOLOE weights not found: {w} (is $YOLO set? default is "
                                 f"{YOLO_DIR}/yoloe/yoloe-11l-seg.pt)")
            # YOLOE looks for mobileclip_blt.ts in the current directory when encoding prompts
            with _cwd(w.parent):
                self.model = YOLOE(str(w))
                self.model.set_classes(list(YOLOE_PROMPTS))
            self.map = {i: YOLOE_PROMPTS[n] for i, n in self.model.names.items()}
            self.name = f"yoloe:{w.name}"
        else:
            from ultralytics import YOLO
            self.model = YOLO(spec)
            names = self.model.names
            if sorted(names.values()) != sorted(CLASSES):
                raise ValueError(f"{spec} has classes {names}, expected {CLASSES}")
            self.map = dict(names)
            self.name = Path(spec).parent.parent.name + "/" + Path(spec).name  # e.g. r1/best.pt
        print(f"[Detector] {self.name} device={self.device} imgsz={self.imgsz}")

    def __call__(self, images):
        """images: list of full-resolution BGR arrays. Returns list of instance lists."""
        res = self.model.predict(images, imgsz=self.imgsz, conf=self.conf, device=self.device,
                                 half=self.half, agnostic_nms=False, max_det=50, verbose=False)
        out = []
        for r in res:
            dets = [(self.map[int(c)], s, b) for c, s, b in
                    zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist())]
            out.append(group_instances(dets, margin=self.margin)[: self.max_inst])
        return out
