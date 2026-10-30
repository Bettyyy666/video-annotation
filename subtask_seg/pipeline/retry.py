"""[D] retry / status per subtask, pure rules on the [A] grasp episodes (+ one VLM yes/no to confirm a miss).

  * missed grasp: a short closure in which the gripper hardly moved and that is followed by the same gripper
    closing again at the same spot (events._confirm_misses), confirmed by the VLM: "did it hold and lift /
    move an object?" p < LIFT_P.
  * re-grasp after placing: after a successful release, the same gripper closes again near the release spot
    inside the same subtask -> retry (task sheet, e.g. cutlery: "Letting go of a utensil and grasping it
    again is a retry"). Rows whose task notes call re-grips normal (straighten / fold) only count real misses.
  * recovery_t = start of the reach before the final attempt: walking back from that grasp's first closed
    frame while the gripper box centre is still moving (at most REACH_MAX_S).
"""
import numpy as np

from . import config as C


def _reach_start(mid_g, c0, f_min, fps):
    v = np.full(len(mid_g), np.nan)
    v[1:] = np.linalg.norm(np.diff(mid_g, axis=0), axis=1)
    t = c0
    lo = max(f_min, c0 - int(C.REACH_MAX_S * fps))
    while t - 1 > lo and not (np.isfinite(v[t - 1]) and v[t - 1] < C.MOTION_SPEED / 2):
        t -= 1
    return t


def judge(eps, row, f0, f1, mid, diag, fps, lifted=None):
    """eps: all episodes; the subtask spans frames (f0, f1]. lifted(e) -> p(object lifted) or None.
    -> status, recovery frame (or None), note (or None), list of debug strings."""
    mine = sorted([e for e in eps if f0 <= e.c0 < f1], key=lambda e: e.c0)
    dbg, bad = [], []

    def dur(e):
        return ((e.onset if e.onset is not None else e.c1) - e.c0) / fps

    for e in mine:
        if not e.failed or dur(e) < C.MIN_MISS_S:   # classifier flicker is not a grasp attempt
            continue
        p = lifted(e) if lifted else None
        if p is not None and p >= C.LIFT_P:
            dbg.append(f"g{e.gripper} {e.c0}-{e.c1}: short closure but VLM p(lifted)={p:.2f} -> not a miss")
            continue
        bad.append((e, "missed grasp"))
        dbg.append(f"g{e.gripper} {e.c0}-{e.c1}: missed grasp" + ("" if p is None else f" (p(lifted)={p:.2f})"))
    if not row.regrasp_ok:
        for k, e in enumerate(mine):
            if e.failed or e.onset is None or dur(e) < C.MIN_CARRY_S:   # a real grasp-and-place ...
                continue
            for e2 in mine[k + 1:]:
                if e2.gripper != e.gripper or dur(e2) < C.MIN_MISS_S:
                    continue
                if (e2.c0 - e.onset) / fps < C.MIN_REGRASP_GAP_S:          # ... then a separate new grasp
                    continue
                if (e2.c0 - e.onset) / fps > 2 * C.RETRY_GAP_S:
                    break
                d = float(np.linalg.norm(mid[min(e2.c0, len(mid) - 1), e.gripper] - mid[e.onset, e.gripper]))
                if np.isfinite(d) and d < C.ASSOC_K * max(diag[e.onset, e.gripper], 1.0):
                    bad.append((e, "re-grasped after placing"))
                    dbg.append(f"g{e.gripper} released at {e.onset}, re-grasped at {e2.c0} nearby")
                break
    bad = [(e, w) for e, w in bad if e is not mine[-1]] if mine else []
    if not bad:
        return "ok", None, None, dbg
    last_bad = max(bad, key=lambda x: x[0].c0)[0]
    final = next((e for e in mine if e.c0 > last_bad.c0), None)
    if final is None:
        return "ok", None, None, dbg + ["retry candidate has no later attempt -> ignored"]
    rec = _reach_start(mid[:, final.gripper], final.c0, max(f0, last_bad.c1), fps)
    n_miss = sum(w == "missed grasp" for _, w in bad)
    n_re = len(bad) - n_miss
    why = ", ".join(x for x in (f"{n_miss} missed grasp(s)" if n_miss else "",
                                f"{n_re} re-grasp(s) after placing" if n_re else "") if x)
    return "suboptimal", rec, f"retry: {why}; final attempt starts at t={rec / fps:.2f}s", dbg
