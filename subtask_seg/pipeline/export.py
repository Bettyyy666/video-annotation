"""Aligned segments -> spec JSON (contiguous subtasks, occurrence numbers, step intervals)."""
from collections import defaultdict

from . import config as C


def _tf(f, fps):
    t = round(f / fps, 3)
    return t, int(round(t * C.FPS_OUT))


def build(episode_id, fps, T, segs, outcome="success", robot=None):
    """segs: time-ordered dicts {slot, f0, f1, status, rec (frame or None), note, detail}; first f0 = 0,
    last f1 = T-1, consecutive (f1 of one = f0 of the next)."""
    occ = defaultdict(int)
    out_subs = []
    for n, s in enumerate(segs, 1):
        r = s["slot"].row
        occ[(r.step_n, r.name)] += 1
        t0, f0 = _tf(s["f0"], fps)
        t1, f1 = _tf(s["f1"], fps)
        rec = None
        if s["status"] == "suboptimal" and s.get("rec") is not None:
            rec = round(s["rec"] / fps, 3)
        out_subs.append({
            "seq": n, "step_n": r.step_n, "name": r.name, "occurrence": occ[(r.step_n, r.name)],
            "object": r.obj, "object_detail": s.get("detail"),
            "t_start": t0, "t_end": t1, "f_start": f0, "f_end": f1,
            "status": s["status"], "recovery_t": rec, "note": s.get("note"),
        })

    # steps: one interval per contiguous run of the same (step, round)
    out_steps, seen = [], set()
    for s, o in zip(segs, out_subs):
        key = (s["slot"].row.step_n, s["slot"].round)
        if out_steps and out_steps[-1]["_key"] == key:
            st = out_steps[-1]
            st["t_end"], st["f_end"] = o["t_end"], o["f_end"]
        else:
            if key in seen:                       # step resumed later -> split interval, say why
                o["note"] = ((o["note"] + "; ") if o["note"] else "") + \
                            "performed out of order; its step is split into several intervals"
            seen.add(key)
            out_steps.append({"_key": key, "step_n": key[0], "step_name": s["slot"].row.step_name,
                              "t_start": o["t_start"], "t_end": o["t_end"],
                              "f_start": o["f_start"], "f_end": o["f_end"]})
    for st in out_steps:
        st.pop("_key")
    return {
        "episode": {"episode_id": episode_id, "robot": robot, "outcome": outcome},
        "steps": out_steps,
        "subtasks": out_subs,
    }
