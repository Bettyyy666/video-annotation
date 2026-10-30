"""Per-frame perception, cached to a pickle.

Gripper boxes and open/closed come from ../gripper (video-annotation/gripper, see its README):
  SAM 2 tracks of the LEFT / RIGHT gripper  -> runs/gripper/tracks/<id>.json
  YOLO11-cls on the gripper crops           -> runs/gripper/pred/<id>.json
from_gripper() turns them into the arrays the event code uses. detect_objects() (YOLO-World + ByteTrack) is
kept for experiments but no longer used: the grammar DP + VLM do not need object boxes, and YOLO-World
called almost every utensil "teaspoon" on 26_sort_cutlery.

Output dict: B (T,2,4) gripper boxes (L, R), P (T,2) p_closed (NaN = unknown / hidden / not tracked),
objs {frame: [[track_id, class, x1, y1, x2, y2, conf], ...]}, prompts, parent, fps, W, H, T.
"""
import json
import pickle

import cv2
import numpy as np

from . import config as C


def build_prompts(vocab):
    """Task-sheet object names + fine-grained names (config.DETAIL_VOCAB) that map back to their parent."""
    prompts, parent = list(vocab), list(range(len(vocab)))
    for i, v in enumerate(vocab):
        for key, details in C.DETAIL_VOCAB.items():
            if key in v.lower():
                for d in details:
                    if d not in prompts and d != v.lower():
                        prompts.append(d)
                        parent.append(i)
    return prompts, parent


def detect_objects(video, vocab, device=None, stride=C.OBJ_STRIDE):
    """YOLO-World + ByteTrack over the whole video (full resolution frames)."""
    from ultralytics import YOLOWorld

    prompts, parent = build_prompts(vocab)
    world = YOLOWorld(C.DET_MODEL) if prompts else None
    if world is not None:
        world.set_classes(prompts)
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or C.FPS_OUT
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    objs, f = {}, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if world is not None and f % stride == 0:
            rr = world.track(frame, persist=True, tracker="bytetrack.yaml", conf=C.DET_CONF,
                             agnostic_nms=True, verbose=False, device=device)[0]
            b = rr.boxes
            if b is not None and len(b):
                ids = b.id.cpu().numpy() if b.id is not None else -np.ones(len(b))
                objs[f] = np.c_[ids, b.cls.cpu().numpy(), b.xyxy.cpu().numpy(), b.conf.cpu().numpy()]
            else:
                objs[f] = np.zeros((0, 7))
        f += 1
    cap.release()
    return dict(video=video, fps=fps, W=W, H=H, T=f, objs=objs, prompts=prompts, parent=parent)


def from_gripper(tracks_json, pred_json, T):
    """tracks: frames[f][L|R] = [x1, y1, x2, y2, score, mark]  (mark 2 = gripper hidden, box carried over)
    pred:   frames[f][L|R] = [p_open, p_closed, p_unknown]
    -> B (T,2,4), P (T,2) p_closed with NaN where unknown / hidden / not tracked."""
    tr = json.load(open(tracks_json))["frames"]
    pr = json.load(open(pred_json))["frames"]
    B = np.full((T, 2, 4), np.nan)
    P = np.full((T, 2), np.nan)
    for f in range(T):
        rec, p = tr.get(str(f), {}), pr.get(str(f), {})
        for s, side in enumerate(("L", "R")):
            v = rec.get(side)
            if v is None:
                continue
            B[f, s] = v[:4]
            q = p.get(side)
            if q is None or (len(v) > 5 and v[5] == 2):
                continue
            po, pc, pu = q
            if pu > max(po, pc):
                continue
            P[f, s] = pc / max(po + pc, 1e-6)
    return B, P


def load(pkl):
    with open(pkl, "rb") as fh:
        return pickle.load(fh)
