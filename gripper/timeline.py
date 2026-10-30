"""Open/closed TIMELINE labels per tracked gripper (used by the label tool's timeline page and by
gripper.state for the classifier dataset).

A unit = one gripper side over one tracked range:  key "<stem>|<a>-<b>|<L or R>".
Labels are change points: [[frame, class, by], ...] sorted by frame; the state of frame f is the
class of the last change point <= f (frames before the first change point are unlabelled).
"bad" = [[f0, f1], ...] frame ranges whose track box is wrong -> never used for training.

<project>/states.json  {key: {"changes": [...], "bad": [...], "done": bool, "t": unix time}}
"""
import json
import os
import threading
import time
from pathlib import Path

import numpy as np

from .common import CLASSES, SIDES

LOCK = threading.Lock()


def unit_key(stem, a, b, side):
    return f"{stem}|{a}-{b}|{side}"


def parse_key(key):
    stem, rng, side = key.split("|")
    a, b = map(int, rng.split("-"))
    return stem, a, b, side


def load_all_tracks(project):
    out = {}
    d = Path(project) / "tracks"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            out[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return out


def units(tracks):
    """[(key, video, a, b, side)] for every tracked range and side."""
    out = []
    for stem, t in tracks.items():
        for a, b in t["ranges"]:
            for side in SIDES:
                out.append((unit_key(stem, a, b, side), t["video"], a, b, side))
    return out


class States:
    def __init__(self, project):
        self.path = Path(project) / "states.json"
        self.data = json.loads(self.path.read_text()) if self.path.exists() else {}

    def get(self, key):
        return self.data.get(key, {"changes": [], "bad": [], "done": False})

    def put(self, key, rec):
        with LOCK:
            rec = dict(rec)
            rec["changes"] = normalize(rec.get("changes", []))
            rec["bad"] = sorted([sorted(map(int, x)) for x in rec.get("bad", [])])
            rec["t"] = time.time()
            if rec.get("done"):
                rec["changes"] = [[f, c, "human"] for f, c, _ in rec["changes"]]
            self.data[key] = rec
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(self.data, indent=1))
            os.replace(tmp, self.path)
            return rec


def normalize(changes):
    """Sort, one change per frame (last wins), drop changes that repeat the previous class."""
    by_f = {}
    for f, c, *by in changes:
        if c in CLASSES:
            by_f[int(f)] = [int(f), c, by[0] if by else "human"]
    out = []
    for f in sorted(by_f):
        if out and out[-1][1] == by_f[f][1]:
            continue
        out.append(by_f[f])
    return out


def dense(changes, bad, a, b):
    """{frame: class} for labelled, not-bad frames in [a, b]."""
    lab = {}
    ch = normalize(changes)
    k, cur = 0, None
    badset = set()
    for f0, f1 in bad:
        badset.update(range(f0, f1 + 1))
    for f in range(a, b + 1):
        while k < len(ch) and ch[k][0] <= f:
            cur = ch[k][1]
            k += 1
        if cur is not None and f not in badset:
            lab[f] = cur
    return lab


def changes_from_pred(pred, a, b, hi=0.65, lo=0.35, min_run=5):
    """Model pre-labels: pred {f: [p_open, p_closed, p_unknown]} -> change points (by 'model').
    Median-smoothed p_open, hysteresis thresholds, runs shorter than min_run frames are absorbed."""
    fs = [f for f in range(a, b + 1) if str(f) in pred]
    if not fs:
        return []
    P = np.array([pred[str(f)] for f in fs], float)
    po = P[:, 0] / np.maximum(P[:, 0] + P[:, 1], 1e-6)
    k = 5
    pad = np.pad(po, k // 2, mode="edge")
    po = np.array([np.median(pad[i:i + k]) for i in range(len(po))])
    lab, cur = [], None
    for i, f in enumerate(fs):
        if P[i, 2] > max(P[i, 0], P[i, 1]):
            s = "gripper_unknown"
        elif po[i] >= hi:
            s = "gripper_open"
        elif po[i] <= lo:
            s = "gripper_closed"
        else:
            s = cur or "gripper_unknown"
        lab.append(s)
        cur = s
    runs = []  # [start_idx, end_idx, state]
    for i, s in enumerate(lab):
        if runs and runs[-1][2] == s:
            runs[-1][1] = i
        else:
            runs.append([i, i, s])
    merged = []
    for r in runs:
        if merged and r[1] - r[0] + 1 < min_run:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return normalize([[fs[r[0]], r[2], "model"] for r in merged])
