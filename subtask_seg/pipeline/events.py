"""Gripper state signal (1 - p_closed) -> grasp episodes (close ... open), each tagged with object, motion and
miss/hit."""
from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np

from . import config as C
from .util import interp_nan as _interp


@dataclass
class Episode:
    gripper: int                 # 0 = left in image, 1 = right
    c0: int                      # first closed frame
    c1: int                      # first frame back in 'open' state (T if still closed at video end)
    onset: Optional[int]         # first frame the fingers START opening (release moment)
    obj: Optional[int] = None    # index into vocab
    tid: Optional[int] = None    # ByteTrack id of that object
    detail: Optional[str] = None
    obj_disp: float = float("nan")
    grip_disp: float = float("nan")
    ap_med: float = float("nan")  # median open-ness while closed (debug only)
    failed: bool = False         # closed on nothing and nothing moved -> missed grasp
    depart: Optional[int] = None # first frame the gripper is away from the object after the grasp
    uncertain: bool = False      # gripper not visible right before the release -> onset is only a bound

    def to_dict(self, fps):
        d = asdict(self)
        d["t_onset"] = None if self.onset is None else round(self.onset / fps, 3)
        return d


def _nan_medfilt(x, k=5):
    """Median filter that leaves NaN runs as NaN and does not smear them into visible frames."""
    out = x.copy()
    h = k // 2
    for t in np.flatnonzero(~np.isnan(x)):
        w = x[max(0, t - h): t + h + 1]
        out[t] = np.nanmedian(w)
    return out


def signals(data):
    """-> open-ness an (T,2) in [0,1] (NaN = unknown), gripper box center mid (T,2,2), box diagonal (T,2)."""
    B, P = data["B"], data["P"]
    diag = np.hypot(B[:, :, 2] - B[:, :, 0], B[:, :, 3] - B[:, :, 1])
    mid = np.stack([(B[:, :, 0] + B[:, :, 2]) / 2, (B[:, :, 1] + B[:, :, 3]) / 2], -1)
    an = np.full(P.shape, np.nan)
    max_gap = int(round(C.MAX_INTERP_S * data["fps"]))
    for s in (0, 1):
        x = _interp(1.0 - P[:, s], max_gap)   # short unknown runs: interpolate; long ones: stay unknown
        if np.isnan(x).all():
            continue
        an[:, s] = _nan_medfilt(x, 5)
        diag[:, s] = _interp(diag[:, s])
        for k in (0, 1):
            mid[:, s, k] = _interp(mid[:, s, k])
    return an, mid, diag


def closed_intervals(an, fps):
    T, s = len(an), False
    closed = np.zeros(T, bool)
    for t in range(T):
        if np.isnan(an[t]):
            closed[t] = s
            continue
        if s and an[t] > C.OPEN_T:
            s = False
        elif not s and an[t] < C.CLOSE_T:
            s = True
        closed[t] = s
    runs, t = [], 0
    while t < T:
        if closed[t]:
            u = t
            while u < T and closed[u]:
                u += 1
            runs.append([t, u])
            t = u
        else:
            t += 1
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < C.MERGE_GAP_S * fps:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [r for r in merged if r[1] - r[0] >= C.MIN_CLOSED_S * fps]


def open_onset(an, c0, c1):
    """First frame where the fingers leave the closed plateau (not where they are fully open)."""
    if c1 >= len(an):
        return None
    n = c1 - c0
    core = an[c0 + n // 5: c0 + max(n // 5 + 1, 4 * n // 5)]
    plateau = float(np.nanmedian(core))
    t = c1
    while t - 1 > c0 and an[t - 1] > plateau + C.ONSET_DELTA:
        t -= 1
    return max(c0 + 1, t - C.ONSET_SHIFT)


class ObjIndex:
    def __init__(self, data):
        self.objs = data["objs"]
        self.frames = np.array(sorted(self.objs))
        self.parent = data["parent"]
        self.prompts = data["prompts"]
        self.nv = len(data["vocab"])

    def at(self, f):
        if len(self.frames) == 0:
            return np.zeros((0, 7))
        i = np.searchsorted(self.frames, f)
        cand = [self.frames[j] for j in (i - 1, i) if 0 <= j < len(self.frames)]
        return self.objs[min(cand, key=lambda g: abs(g - f))]

    def find_tid(self, tid, f, back=25):
        if tid is None:
            return None
        for g in self.frames[(self.frames <= f) & (self.frames >= f - back)][::-1]:
            hit = self.objs[g][self.objs[g][:, 0] == tid]
            if len(hit):
                return hit[0, 2:6]
        return None

    def instance_counts(self, T, fps):
        """Objects of each class visible at the start (90th pct of per-frame count over the first 10%)."""
        lim = max(int(0.1 * T), int(2 * fps))
        fr = self.frames[self.frames < lim]
        out = {}
        for v in range(self.nv):
            cnt = [int(sum(self.parent[int(c)] == v for c in self.objs[g][:, 1])) for g in fr]
            if cnt:
                out[v] = int(round(np.percentile(cnt, 90)))
        return out


def _box_dist(p, box):
    dx = max(box[0] - p[0], 0, p[0] - box[2])
    dy = max(box[1] - p[1], 0, p[1] - box[3])
    return float(np.hypot(dx, dy))


def _center(box):
    return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])


def extract(data):
    fps, T = data["fps"], data["T"]
    an, mid, diag = signals(data)
    idx = ObjIndex(data)
    eps = []
    for g in (0, 1):
        if np.isnan(an[:, g]).all():
            continue
        for c0, c1 in closed_intervals(an[:, g], fps):
            if C.PARK_K > 0 and _parked(mid[:, g], diag[:, g], c0, c1, fps):
                continue                             # an idle arm: open/close flicker is not a grasp
            e = Episode(g, c0, c1, open_onset(an[:, g], c0, c1))
            if e.onset is not None:                  # state unknown right before the release?
                w = an[max(c0, e.onset - C.TOL_FRAMES): e.onset + 1, g]
                e.uncertain = bool(np.isnan(w).any())
            end = e.onset if e.onset is not None else min(c1, T)
            fa = min(c0 + 5, end - 1, T - 1)
            fb = min(max(fa, end - 2), T - 1)
            m, dg = mid[fa, g], diag[fa, g]
            best, bd = None, np.inf
            for row in idx.at(fa):
                d = _box_dist(m, row[2:6])
                if d < bd:
                    best, bd = row, d
            if best is not None and bd < C.ASSOC_K * dg:
                cls = int(best[1])
                e.obj = idx.parent[cls]
                e.tid = int(best[0]) if best[0] >= 0 else None
                e.detail = idx.prompts[cls] if cls >= idx.nv else None
                box_b = idx.find_tid(e.tid, fb)
                if box_b is not None:
                    e.obj_disp = float(np.linalg.norm(_center(box_b) - _center(best[2:6])))
                e.depart = _depart(e, mid[:, g], diag[:, g], idx, best[2:6], fb, T)
            e.grip_disp = float(np.linalg.norm(mid[fb, g] - mid[fa, g]))
            e.ap_med = float(np.nanmedian(an[c0:min(c1, T), g]))
            carried = (e.obj_disp > C.D_OBJ) or (np.isnan(e.obj_disp) and e.grip_disp > C.D_GRIP)
            e.failed = ((c1 - c0) / fps < C.FAIL_MAX_S) and not carried   # candidate, confirmed below
            eps.append(e)
    _confirm_misses(eps, mid, diag, fps)
    eps.sort(key=lambda e: (e.c0, e.gripper))
    return eps, an, idx


def _parked(mid_g, diag_g, c0, c1, fps):
    a, b = max(0, c0 - int(C.PARK_S * fps)), min(len(mid_g), c1 + int(C.PARK_S * fps))
    w = mid_g[a:b]
    w = w[np.all(np.isfinite(w), axis=1)]
    if len(w) < 2:
        return False
    span = float(np.max(np.linalg.norm(w - np.median(w, axis=0), axis=1)))
    d = np.nanmedian(diag_g[a:b])
    return bool(np.isfinite(d) and span < C.PARK_K * d)


def _confirm_misses(eps, mid, diag, fps):
    """A missed grasp is a short closure that moved nothing AND is followed by the same gripper closing again
    at the same spot (the retry). Without the re-approach it is a hold / press / steadying closure, not a miss:
    the state classifier cannot tell an empty closed gripper from one pinching a thin object."""
    for g in (0, 1):
        mine = sorted((e for e in eps if e.gripper == g), key=lambda e: e.c0)
        for i, e in enumerate(mine):
            if not e.failed:
                continue
            nxt = mine[i + 1] if i + 1 < len(mine) else None
            ok = nxt is not None and (nxt.c0 - e.c1) / fps <= C.RETRY_GAP_S
            if ok:
                a, b = mid[min(e.c0, len(mid) - 1), g], mid[min(nxt.c0, len(mid) - 1), g]
                same_obj = e.tid is not None and e.tid == nxt.tid
                ok = same_obj or float(np.linalg.norm(a - b)) < C.ASSOC_K * diag[e.c0, g]
            e.failed = bool(ok)


def _depart(e, mid_g, diag_g, idx, box0, f_from, T):
    """For subtasks that end without a release: first frame the gripper stays away from the object."""
    box, away = box0, 0
    for f in range(f_from, min(T, f_from + 500)):   # look at most 10 s ahead
        nb = idx.find_tid(e.tid, f, back=4) if e.tid is not None else None
        if nb is not None:
            box = nb
        if _box_dist(mid_g[f], box) > C.ASSOC_K * diag_g[f]:
            away += 1
            if away >= C.DEPART_HOLD:
                return f - C.DEPART_HOLD + 1
        else:
            away = 0
    return None
