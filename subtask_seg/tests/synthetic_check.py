"""Logic check without a GPU, videos or VLM: fake gripper signals with known grasps + fake VLM row scores ->
events / grammar DP / retry / export / validate; assert boundaries, retry and counts come out as constructed.

    python tests/synthetic_check.py
"""
import json
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import align, events, export, retry, tasksheet, validate  # noqa: E402

FPS, T = 50.0, 2500
SHEET = {"version": "test", "tasks": [{"task": "synthetic", "title": "Synthetic", "notes": [], "steps": [
    {"n": 1, "title": "Move the cup"}, {"n": 2, "title": "Put the blocks in the box"},
    {"n": 3, "title": "Clean up"}, {"n": 4, "title": "End the episode"}],
    "subtasks": [
    {"step": 1, "subtask": "Place the cup on the tray", "object": "cup", "count": "1",
     "ends_when": "The cup is released on the tray"},
    {"step": 2, "subtask": "Put one block in the box", "object": "block", "count": "per block",
     "ends_when": "The block is released in the box"},
    {"step": 3, "subtask": "Wipe the table", "object": "cloth", "count": "0–1, only if something spilled",
     "ends_when": "The cloth leaves the table"},
    {"step": 4, "subtask": "Return both arms home and end the episode", "object": "–", "count": "1",
     "ends_when": "Last frame"}]}]}

# (gripper, close_start, close_end, object_id, success)   object ids: 1 cup, 2..4 blocks
EVENTS = [(0, 100, 300, 1, True), (0, 400, 480, 2, False), (0, 600, 800, 2, True),
          (0, 900, 1100, 3, True), (1, 1200, 1400, 4, True)]
OBJ0 = {1: (300, 300), 2: (300, 500), 3: (400, 500), 4: (900, 500)}
CLS = {1: 2, 2: 0, 3: 0, 4: 0}          # vocab is sorted: block, cloth, cup
RAMP = 10


def make_data():
    ap = np.full((T, 2), 0.6)
    pos = {k: np.array(v, float) for k, v in OBJ0.items()}
    obj_track = {k: np.zeros((T, 2)) for k in pos}
    mid = np.zeros((T, 2, 2))
    mid[:, 0], mid[:, 1] = (200, 200), (1000, 200)
    truth = []
    for g, a, b, oid, ok in EVENTS:
        closed = 0.02
        ap[a - RAMP:a, g] = np.linspace(0.6, closed, RAMP)
        ap[a:b, g] = closed
        ap[b:b + RAMP, g] = np.linspace(closed, 0.6, RAMP)
        if ok:
            truth.append(b)                           # release onset ≈ first frame leaving the plateau
    for t in range(T):
        for g, a, b, oid, ok in EVENTS:
            if a - 40 <= t < b + RAMP:
                if ok and t >= a:                     # carry the object 300 px to the right
                    frac = (t - a) / (b - a) if t < b else 1.0
                    pos[oid] = np.array(OBJ0[oid]) + [300 * frac, -100 * frac]
                mid[t, g] = pos[oid]
        for k in pos:
            obj_track[k][t] = pos[k]
    B = np.concatenate([mid - 40.0, mid + 40.0], -1)  # (T,2,4) gripper boxes
    P = np.clip(1.0 - ap / 0.6, 0, 1) * 0.9 + 0.05   # classifier output: open 0.05 ... closed ~0.95
    P[700:704, 0] = np.nan                            # a few 'unknown' frames
    objs = {}
    for t in range(0, T, 2):
        objs[t] = np.array([[k, CLS[k], *(obj_track[k][t] - 30), *(obj_track[k][t] + 30), 0.6] for k in pos])
    data = dict(video="synthetic", fps=FPS, W=1280, H=720, T=T, B=B, P=P, objs=objs,
                vocab=["block", "cloth", "cup"], prompts=["block", "cloth", "cup"], parent=[0, 1, 2])
    return data, truth


def fake_clips(rows, bounds):
    """VLM stand-in: every second a 2-s clip; the row whose true segment contains the clip centre gets
    p=0.7, the rest share 0.3 (plus noise)."""
    rng = np.random.default_rng(0)
    clips = []
    for f0 in range(0, T - 100, 50):
        c = f0 + 50
        true = next(r for r, (a, b) in bounds.items() if a < c <= b)
        p = np.full(len(rows), 0.3 / (len(rows) - 1)) + rng.uniform(0, 0.02, len(rows))
        p[true] = 0.7
        p /= p.sum()
        clips.append({"f0": f0, "f1": f0 + 100, "center": c, "lp": list(np.log(p))})
    return clips


def main():
    data, truth = make_data()
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(SHEET, fh)
    task, steps, vocab, repeats = tasksheet.load(fh.name, "99_synthetic")
    eps, an, idx = events.extract(data)
    _, mid, diag = events.signals(data)
    rows = [r for st in steps for r in st.rows]
    # true row per frame range (row index into `rows`): cup, block, cloth (never), tail
    bounds = {0: (-1, truth[0]), 1: (truth[0], truth[3]), 3: (truth[3], T)}
    clips = fake_clips(rows, bounds)
    counts = {"block": [0, 0, 0.05, 0.9, 0.05, 0, 0, 0, 0, 0]}
    cond_p = {"Wipe the table": 0.05}
    n_inst = {"block": 3}
    slots = tasksheet.expand(steps, repeats, n_inst)
    row_index = [next(k for k, r in enumerate(rows) if r is s.row) for s in slots]
    emis = align.Emission(clips, len(rows))
    cand, kinds = align.candidates(eps, data["B"], T, FPS)
    segs, cost, dbg = align.align(slots, row_index, emis, cand, kinds, FPS, counts, cond_p)
    out = []
    for j, f0, f1 in segs:
        row = slots[j].row
        st, rec, note, why = ("ok", None, None, []) if tasksheet.is_tail(row) else \
            retry.judge(eps, row, f0, f1, mid, diag, FPS, None)
        out.append({"slot": slots[j], "f0": f0, "f1": f1, "status": st, "rec": rec, "note": note, "detail": None})
    res = export.build("99_synthetic", FPS, T, out)
    issues = validate.check(res, steps, n_inst, {}, T, [e.onset for e in eps if e.onset is not None],
                            [e for e in eps if e.failed], FPS)
    print(json.dumps(res, indent=1))
    print("episodes:", [(e.gripper, e.c0, e.onset, e.failed) for e in eps], "| skipped:", dbg["skipped"])
    print("issues:", issues)

    subs = res["subtasks"]
    assert [s["name"] for s in subs] == ["Place the cup on the tray"] + ["Put one block in the box"] * 3 + \
        ["Return both arms home and end the episode"], subs
    assert [s["occurrence"] for s in subs] == [1, 1, 2, 3, 1]
    assert subs[1]["status"] == "suboptimal" and subs[1]["recovery_t"] is not None
    assert all(s["status"] == "ok" for i, s in enumerate(subs) if i != 1)
    for s, b in zip(subs[:-1], truth):              # every release-ended boundary, incl. the last block
        assert abs(s["f_end"] - b) <= 3, (s["f_end"], b)
    assert subs[0]["f_start"] == 0 and subs[-1]["f_end"] == T - 1
    assert [st["step_n"] for st in res["steps"]] == [1, 2, 4]
    assert subs[-1]["object"] is None
    assert res["episode"]["outcome"] == "success"
    assert not issues, issues

    # a short closure that moves nothing but is NOT followed by a re-grasp is a hold/press, not a miss
    hold = events.Episode(1, 1700, 1760, 1758, failed=True)
    events._confirm_misses([hold], np.zeros((T, 2, 2)), np.full((T, 2), 100.0), FPS)
    assert not hold.failed
    print("\nOK: boundaries within 3 frames of construction, retry detected, conditional row skipped, "
          "counts and steps correct")


if __name__ == "__main__":
    main()
