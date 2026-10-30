"""The five pre-submission checks from the spec + format checks. Returns a list of human-readable issues."""
from collections import Counter

from . import config as C


def check(result, steps, n_inst_named, rounds_done, T, onsets, failed_eps, fps, uncertain_eps=()):
    issues = []
    subs, stps = result["subtasks"], result["steps"]
    rows = {(r.step_n, r.name): r for st in steps for r in st.rows}
    step_names = {st.step_n: st.step_name for st in steps}
    objs = {r.obj for r in rows.values() if r.obj}

    # format: tiling from frame 0 to the last frame
    if subs[0]["f_start"] != 0:
        issues.append("first subtask does not start at frame 0")
    last = int(round(round((T - 1) / fps, 3) * C.FPS_OUT))
    if subs[-1]["f_end"] != last:
        issues.append(f"last subtask ends at {subs[-1]['f_end']}, video ends at {last}")
    for a, b in zip(subs, subs[1:]):
        if a["t_end"] != b["t_start"]:
            issues.append(f"gap/overlap between seq {a['seq']} and {b['seq']}")
        if b["t_end"] <= b["t_start"]:
            issues.append(f"seq {b['seq']} has zero/negative length")

    # 1. counts (fixed-count rows: totals; per-item rows: objects detected in the scene)
    cnt = Counter((s["step_n"], s["name"]) for s in subs)
    done = result["episode"]["outcome"] == "success"
    for key, r in rows.items():
        got = cnt.get(key, 0)
        nr = rounds_done.get(r.step_n, 0)
        if r.count is not None:                         # task-sheet counts are totals per episode
            lo, hi = r.count
            if (done and got < lo) or got > hi:
                issues.append(f"[1] '{r.name}': {got} occurrences, task sheet expects {lo}-{hi}")
        elif not r.per_item_cond:
            n = n_inst_named.get(r.obj)
            if done and n is not None and got != n and nr <= 1:
                issues.append(f"[1] '{r.name}': {got} occurrences, {n} '{r.obj}' detected in scene (check by eye)")

    # 2. step attachment and non-overlap
    for s in subs:
        if (s["step_n"], s["name"]) not in rows:
            issues.append(f"[2] seq {s['seq']} '{s['name']}' not under step {s['step_n']} in the task sheet")
    iv = sorted((st["f_start"], st["f_end"]) for st in stps)
    for a, b in zip(iv, iv[1:]):
        if b[0] < a[1]:
            issues.append(f"[2] step intervals overlap: {a} {b}")

    # 3. names are verbatim
    for st in stps:
        if step_names.get(st["step_n"]) != st["step_name"]:
            issues.append(f"[3] step_name '{st['step_name']}' not verbatim")
    for s in subs:
        if s["object"] is not None and s["object"] not in objs:
            issues.append(f"[3] object '{s['object']}' not in task sheet")

    # 4. release-ended subtasks: a gripper starts opening within ±0.5 s of t_end
    for s in subs[:-1]:
        r = rows.get((s["step_n"], s["name"]))
        if r is None or not r.release:
            continue
        f = s["t_end"] * fps
        if not any(abs(o - f) <= C.TOL_FRAMES for o in onsets):
            issues.append(f"[4] seq {s['seq']} ends at {s['t_end']}s with no gripper opening within 0.5 s")

    # 5. every missed grasp (closed, nothing lifted, reopened) sits in a suboptimal/failed subtask
    for e in failed_eps:
        t = e.c0 / fps
        s = next((s for s in subs if s["t_start"] <= t < s["t_end"]), subs[-1])
        r = rows.get((s["step_n"], s["name"]))
        if s["status"] == "ok" and not (r and r.regrasp_ok):
            issues.append(f"[5] missed grasp at {t:.2f}s (gripper {e.gripper}) inside seq {s['seq']} marked ok "
                          f"(is it a retry, or a hold/press?)")
    # gripper hidden right before a release: boundary may be off, look at it by eye
    for e in uncertain_eps:
        if any(abs(s["f_end"] - round(e.onset / fps * C.FPS_OUT)) <= C.TOL_FRAMES for s in subs[:-1]):
            issues.append(f"[occluded] gripper {e.gripper} not visible before its release at {e.onset / fps:.2f}s; "
                          f"boundary there is an upper bound")
    return issues
