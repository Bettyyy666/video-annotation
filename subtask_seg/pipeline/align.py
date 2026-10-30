"""[C] Task sheet as a grammar + segmental DP over candidate boundaries (no training).

Expected sequence = the task sheet rows in order, each row repeated `count` times (tasksheet.expand); every
occurrence is a slot. Slots can be skipped at a cost (soft counts):
  * occurrence q of a fixed-count row: C_SKIP_MAND * P(at least q objects present)   (VLM count question;
    1 when no count is available)  -> 26_sort_cutlery: only one dinner spoon in the scene, so skipping the
    second costs almost nothing, while a real knife cannot be skipped cheaply;
  * occurrences beyond the minimum: C_SKIP_OPT;
  * conditional rows ("only if ..."): using costs C_COND * (1 - p_yes), skipping C_COND * p_yes, p_yes from
    a VLM yes/no question on the first frames;
  * the final "Last frame" row always ends at the last frame.
Candidate END frames of a subtask:
  * gripper "starts opening" events from [A] (both grippers) - the spec's release moment;
  * "gripper starts moving after being still" (box centre speed) - for rows that do not end with a release;
  * a 1-s grid as a costly fallback, so one missed release does not shift every later subtask.
Segment score = sum over the VLM clips centred inside the segment of the row's log-probability (with each
option's mean over the video removed). DP minimises  -VLM_W * score + boundary costs + skip costs.
Rows stay in task-sheet order (both required videos follow it); a row done out of order would be labelled
as the row the DP fits there - a known limitation.
"""
import bisect

import numpy as np

from . import config as C
from .tasksheet import is_tail


def motion_starts(B, fps):
    """Frames where a gripper box centre starts moving after being still for MOTION_STILL_S."""
    out = []
    still_need = int(round(C.MOTION_STILL_S * fps))
    for s in (0, 1):
        mid = (B[:, s, :2] + B[:, s, 2:]) / 2
        if np.isnan(mid).all():
            continue
        v = np.full(len(mid), np.nan)
        v[2:-2] = np.linalg.norm(mid[4:] - mid[:-4], axis=1) / 4
        still = 0
        for t in range(len(v)):
            if np.isnan(v[t]):
                still = 0
                continue
            if v[t] < C.MOTION_SPEED / 2:
                still += 1
            elif v[t] > C.MOTION_SPEED:
                if still >= still_need:
                    out.append(t)
                still = 0
    return sorted(set(out))


def candidates(eps, B, T, fps):
    """-> sorted frames, and per frame the set of kinds {'release', 'motion', 'grid'}."""
    kinds = {}
    for e in eps:
        if e.onset is not None and 0 < e.onset < T - 1:   # a probable missed grasp cannot end a subtask
            kinds.setdefault(int(e.onset), set()).add("release_miss" if e.failed else "release")
    for f in motion_starts(B, fps):
        if 0 < f < T - 1:
            kinds.setdefault(int(f), set()).add("motion")
    step = int(round(fps))
    for f in range(step, T - 1, step):
        kinds.setdefault(f, set()).add("grid")
    kinds.setdefault(T - 1, set()).add("end")
    frames = sorted(kinds)
    return [0] + frames, kinds


def end_cost(row, kinds_at):
    if is_tail(row):
        return 0.0 if "end" in kinds_at else np.inf
    if "end" in kinds_at:
        return 0.0
    if row.release:
        if "release" in kinds_at:
            return C.C_CAND_RELEASE
        if "motion" in kinds_at or "release_miss" in kinds_at:
            return C.C_CAND_MOTION_FOR_REL
        return C.C_CAND_GRID
    if "motion" in kinds_at:
        return C.C_CAND_MOTION
    if "release" in kinds_at or "release_miss" in kinds_at:
        return C.C_CAND_RELEASE_FOR_MOVE
    return C.C_CAND_GRID


class Emission:
    """Prefix sums of centred VLM log-probs over clip centres; score(row, a, b) for frames (a, b]."""

    def __init__(self, clips, n_rows):
        if clips:
            lp = np.array([c["lp"] for c in clips], float) / C.VLM_TEMP
            lp = lp - np.logaddexp.reduce(lp, axis=1, keepdims=True)   # renormalise after tempering
            lp = np.maximum(lp, C.VLM_FLOOR)             # one confident wrong clip cannot veto a row
            if C.VLM_CENTER:
                lp = lp - lp.mean(0, keepdims=True)      # remove each option's prior over the video
            self.centres = [c["center"] for c in clips]
        else:
            lp = np.zeros((0, n_rows))
            self.centres = []
        self.pref = np.vstack([np.zeros((1, n_rows)), np.cumsum(lp, 0)])

    def score(self, r, a, b):
        i = bisect.bisect_right(self.centres, a)
        j = bisect.bisect_right(self.centres, b)
        return float(self.pref[j, r] - self.pref[i, r])


def slot_costs(slots, counts, cond_p):
    """-> use cost, skip cost per slot. counts: {obj: [p(0)..p(9)]}, cond_p: {row name: p_yes}.
    With a VLM count for the row's object, occurrence q costs C_SKIP_MAND * P(fewer than q objects) to
    use and C_SKIP_MAND * P(at least q) to skip (soft count); without one, the task sheet count decides."""
    use, skip, seen = [], [], {}
    for s in slots:
        r = s.row
        q = seen.get(id(r), 0) + 1                        # occurrence number of this row
        seen[id(r)] = q
        if is_tail(r):
            use.append(0.0)
            skip.append(np.inf)
            continue
        if r.name in cond_p:
            p = cond_p[r.name]
            use.append(C.C_COND * (1 - p))
            skip.append(C.C_COND * p)
            continue
        if r.obj in counts:
            pc = np.array(counts[r.obj], float)
            p_ge = float(pc[q:].sum() / max(pc.sum(), 1e-9))       # P(at least q objects present)
            use.append(C.C_SKIP_MAND * (1 - p_ge))
            skip.append(C.C_SKIP_MAND * p_ge)
        elif s.mandatory:
            use.append(0.0)
            skip.append(C.C_SKIP_MAND)
        else:
            use.append(C.C_EXTRA_OCC)
            skip.append(C.C_SKIP_OPT)
    return use, skip


def align(slots, row_index, emis, cand, kinds, fps, counts=None, cond_p=None):
    """-> list of (slot index, f_start, f_end) in time order, total cost, debug dict."""
    counts, cond_p = counts or {}, cond_p or {}
    use, skip = slot_costs(slots, counts, cond_p)
    M, N = len(slots), len(cand)
    cand_a = np.asarray(cand, float)
    ci = np.array([bisect.bisect_right(emis.centres, f) for f in cand])     # clip-prefix index per candidate
    min_len = C.MIN_SEG_S * fps
    INF = np.inf
    dp = np.full((M + 1, N), INF)
    bp = {}
    dp[0, 0] = 0.0
    for j in range(M):
        r = row_index[j]
        row = slots[j].row
        tail = is_tail(row)
        ecost = np.array([end_cost(row, kinds.get(f, set())) for f in cand])
        ecost[0] = INF
        pref_r = emis.pref[:, r]
        for i in range(N):
            base = dp[j, i]
            if base == INF:
                continue
            if base + skip[j] < dp[j + 1, i]:                        # skip slot j
                dp[j + 1, i] = base + skip[j]
                bp[(j + 1, i)] = (i, False)
            ks = np.arange(N - 1, N) if tail else np.arange(i + 1, N)
            ks = ks[ks > i]
            if not len(ks):
                continue
            ok = (cand_a[ks] - cand_a[i] >= min_len) | (ks == N - 1)
            ks = ks[ok]
            if not len(ks):
                continue
            score = pref_r[ci[ks]] - pref_r[ci[i]]
            c = base + use[j] + ecost[ks] - C.VLM_W * score
            better = c < dp[j + 1, ks]
            for k, v in zip(ks[better], c[better]):
                dp[j + 1, k] = v
                bp[(j + 1, int(k))] = (i, True)
    if not np.isfinite(dp[M, N - 1]):
        raise RuntimeError("alignment failed: no path reaches the last frame")
    segs, j, k = [], M, N - 1
    while j > 0:
        i, used = bp[(j, k)]
        if used:
            segs.append((j - 1, cand[i], cand[k]))
        j, k = j - 1, i
    segs.reverse()
    used_slots = {s for s, _, _ in segs}
    skipped = [{"slot": j, "row": slots[j].row.name, "skip_cost": round(float(skip[j]), 3)}
               for j in range(M) if j not in used_slots]
    return segs, float(dp[M, N - 1]), {"skipped": skipped, "use": use, "skip": skip}
