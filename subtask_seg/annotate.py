"""One command per video (from video-annotation/subtask_seg; the video's grippers must already be tracked and
classified by ../gripper, see README):

    python annotate.py ../data/12_fold_four_towels.mp4 --onset-shift -7

  [A] gripper boxes + open/closed (../runs/gripper/{tracks,pred}/<id>.json) -> grasp episodes, release frames
  [B] Qwen3-VL-8B log-probs: which task-sheet row each 2-s clip shows; yes/no for conditional rows and
      missed grasps; counts of objects at the start          (cached in results/vlm/<id>.json)
  [C] task sheet as a grammar, DP over candidate boundaries (releases / motion starts / 1-s fallback)
  [D] retry / status rules per subtask

writes results/<id>.json (submission) and results/debug/<id>_events.json (episodes, candidates, chosen
segments and their costs, VLM summary, retry reasoning, self-check issues).
"""
import argparse
import json
import os
import re
import time

import numpy as np

from pipeline import align, events, export, perception, retry, tasksheet, validate, vlm
from pipeline import config as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--task-sheets", default="../data/task_sheets.json")
    ap.add_argument("--gripper-dir", default="../runs/gripper",
                    help="gripper package output: tracks/<id>.json (SAM 2 boxes) and pred/<id>.json (open/closed)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--robot", default="", help="robot name for the JSON ('' -> null)")
    ap.add_argument("--onset-shift", type=int, default=None,
                    help="frames subtracted from every detected release (classifier lag, tools/measure_lag.py)")
    ap.add_argument("--vlm-model", default=C.VLM_MODEL)
    ap.add_argument("--vlm-width", type=int, default=C.VLM_WIDTH)
    ap.add_argument("--no-vlm", action="store_true", help="no VLM at all (uniform row scores; for debugging)")
    ap.add_argument("--no-vlm-compute", action="store_true", help="use cached VLM answers only (no GPU)")
    a = ap.parse_args()
    if a.onset_shift is not None:
        C.ONSET_SHIFT = a.onset_shift

    ep_id = os.path.splitext(os.path.basename(a.video))[0]
    for d in ("debug", "vlm"):
        os.makedirs(os.path.join(a.out, d), exist_ok=True)
    t0 = time.time()

    # ---- [A] gripper
    task, steps, vocab, repeats = tasksheet.load(a.task_sheets, ep_id)
    tj = os.path.join(a.gripper_dir, "tracks", ep_id + ".json")
    pj = os.path.join(a.gripper_dir, "pred", ep_id + ".json")
    for p in (tj, pj):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p}: run python -m gripper.track video / gripper.state predict for {ep_id}")
    tr = json.load(open(tj))
    fps = float(tr.get("fps") or 50.0)
    T = max(int(f) for f in tr["frames"]) + 1
    data = {"fps": fps, "T": T, "W": tr.get("w", 1280), "H": tr.get("h", 720), "objs": {}, "prompts": [],
            "parent": [], "vocab": list(vocab)}
    data["B"], data["P"] = perception.from_gripper(tj, pj, T)
    eps, an, idx = events.extract(data)
    _, mid, diag = events.signals(data)

    # ---- [B] VLM
    rows = [r for st in steps for r in st.rows]
    vc = None if a.no_vlm else vlm.VlmCache(os.path.join(a.out, "vlm", ep_id + ".json"), a.video, a.vlm_model,
                                             a.vlm_width, allow_compute=not a.no_vlm_compute)
    clips = vc.rows(task.get("title", task["task"]), rows) if vc else []
    cond_p, counts = {}, {}
    for r in rows:
        cond = re.search(r"only if (.+)$", r.count_text, re.I)
        if cond and vc:
            q = cond.group(1).strip().rstrip(".")
            cond_p[r.name] = vc.yes(f"cond:{r.name}", lambda s, c, q=q: vlm.ask_condition(s, c, q))
        several = (r.count and r.count[1] > 1) or (r.count is None and not r.per_item_cond)
        if vc and r.obj and several and r.obj not in counts:   # rows done once per object: soft count
            counts[r.obj] = vc.count(r.obj)
    n_inst = {o: int(np.argmax(p)) for o, p in counts.items()}

    # ---- [C] grammar DP
    slots = tasksheet.expand(steps, repeats, n_inst)
    row_index = [next(k for k, r in enumerate(rows) if r is s.row) for s in slots]
    emis = align.Emission(clips, len(rows))
    cand, kinds = align.candidates(eps, data["B"], T, fps)
    segs, cost, dbg = align.align(slots, row_index, emis, cand, kinds, fps, counts, cond_p)

    # ---- [D] retry / status
    def box_at(g):
        return lambda f: data["B"][min(max(f, 0), T - 1), g]

    def lifted(e):
        if vc is None:
            return None
        return vc.yes(f"lift:{e.gripper}:{e.c0}:{e.c1}",
                      lambda s, c: vlm.ask_lifted(s, c, box_at(e.gripper), e.c0, e.c1,
                                                  "left" if e.gripper == 0 else "right"), optional=True)

    out_segs, seg_dbg, misses = [], [], []
    for j, f0, f1 in segs:
        row = slots[j].row
        if tasksheet.is_tail(row):
            st, rec, note, why = "ok", None, None, []
        else:
            st, rec, note, why = retry.judge(eps, row, f0, f1, mid, diag, fps, lifted)
        misses += [w for w in why if "missed grasp" in w and "not a miss" not in w]
        kind = sorted(kinds.get(f1, {"end"}))
        out_segs.append({"slot": slots[j], "f0": f0, "f1": f1, "status": st, "rec": rec, "note": note,
                         "detail": None})
        seg_dbg.append({"row": row.name, "f0": f0, "f1": f1, "end_kind": kind,
                        "vlm_score": round(emis.score(row_index[j], f0, f1), 2), "status": st, "retry": why})

    # outcome: a row the task sheet requires (and the VLM count supports) is missing after the last subtask
    used = [j for j, _, _ in segs if not tasksheet.is_tail(slots[j].row)]
    last_used = max(used) if used else -1
    trailing = [s for s in dbg["skipped"] if s["slot"] > last_used and s["skip_cost"] >= C.C_SKIP_MAND / 2]
    outcome = "failed" if trailing else "success"
    result = export.build(ep_id, fps, T, out_segs, outcome, robot=a.robot or None)
    with open(os.path.join(a.out, ep_id + ".json"), "w") as fh:
        json.dump(result, fh, indent=2, ensure_ascii=False)

    onsets = [e.onset for e in eps if e.onset is not None]
    rounds_done = {}
    for j, _, _ in segs:
        rounds_done.setdefault(slots[j].row.step_n, set()).add(slots[j].round)
    rounds_done = {k: len(v) for k, v in rounds_done.items()}
    confirmed = [e for e in eps if e.failed and any(f"g{e.gripper} {e.c0}-{e.c1}: missed" in m for m in misses)]
    issues = validate.check(result, steps, n_inst, rounds_done, T, onsets, confirmed, fps,
                            [e for e in eps if e.uncertain])
    best = [rows[int(np.argmax(c["lp"]))].name[:40] for c in clips]
    with open(os.path.join(a.out, "debug", ep_id + "_events.json"), "w") as fh:
        json.dump({"fps": fps, "T": T, "onset_shift": C.ONSET_SHIFT, "dp_cost": cost,
                   "vlm_counts": counts, "vlm_conditions": cond_p,
                   "vlm_top_row_per_clip": [[c["center"], b] for c, b in zip(clips, best)],
                   "candidates": {str(f): sorted(kinds.get(f, [])) for f in cand},
                   "segments": seg_dbg, "skipped_slots": dbg["skipped"],
                   "episodes": [e.to_dict(fps) for e in eps], "issues": issues}, fh, indent=1, default=float)

    print(f"{ep_id}: {len(eps)} grasp episodes, {len(cand)} candidate boundaries, {len(clips)} VLM clips, "
          f"{len(result['subtasks'])} subtasks, outcome={outcome}  | {time.time() - t0:.0f}s")
    for s in issues:
        print("  CHECK", s)


if __name__ == "__main__":
    main()
