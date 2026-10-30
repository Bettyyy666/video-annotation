"""How late does the pipeline's release frame come out, compared with the human "fingers start to open"
frames on the gripper timeline labels?  -> value for config.ONSET_SHIFT.

For every DONE timeline unit (video-annotation/runs/gripper/states.json), take the model's per-frame
prediction for that gripper (runs/gripper/pred/<video>.json), run the SAME signal processing and release
detection as the pipeline (events.signals / closed_intervals / open_onset, with ONSET_SHIFT = 0), and match
every human closed->open change point to the nearest detected release within ±--win frames.

Units whose segment is in the classifier's validation split are reported separately: those were not
trained on, so their lag is the honest one; use their median for ONSET_SHIFT.

    cd video-annotation/subtask_seg
    python tools/measure_lag.py --gripper-dir ../runs/gripper
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline import config as C  # noqa: E402
from pipeline import events  # noqa: E402


def p_closed_series(pred, track, a, b, side):
    out = np.full(b - a + 1, np.nan)
    for f in range(a, b + 1):
        q = pred.get(str(f), {}).get(side)
        v = track.get(str(f), {}).get(side)
        if q is None or v is None or (len(v) > 5 and v[5] == 2):
            continue
        po, pc, pu = q
        if pu > max(po, pc):
            continue
        out[f - a] = pc / max(po + pc, 1e-6)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gripper-dir", default="../runs/gripper")
    ap.add_argument("--win", type=int, default=50, help="max |frames| between a human and a detected release")
    ap.add_argument("--fps", type=float, default=50.0)
    a = ap.parse_args()
    C.ONSET_SHIFT = 0

    gd = a.gripper_dir
    states = json.load(open(os.path.join(gd, "states.json")))
    segments = json.load(open(os.path.join(gd, "segments.json")))
    try:  # same validation split as gripper.state export
        sys.path.insert(0, os.path.abspath(os.path.join(gd, "..", "..")))
        from gripper.state import val_segments
        from gripper.common import Store
        vseg = val_segments(Store(gd), set(segments), 0.2, 0)
    except Exception as e:  # noqa: BLE001
        print(f"(could not load the classifier's val split: {e}; reporting all units together)")
        vseg = {}

    cache = {}
    res = {"val": [], "train": []}
    miss = {"val": 0, "train": 0}
    extra = {"val": 0, "train": 0}
    for key, st in sorted(states.items()):
        if not st.get("done"):
            continue
        stem, rng, side = key.split("|")
        f0, f1 = map(int, rng.split("-"))
        if stem not in cache:
            pp = os.path.join(gd, "pred", stem + ".json")
            tp = os.path.join(gd, "tracks", stem + ".json")
            if not (os.path.exists(pp) and os.path.exists(tp)):
                print(f"{key}: no prediction / track, skipped")
                continue
            cache[stem] = (json.load(open(pp))["frames"], json.load(open(tp))["frames"])
        pred, track = cache[stem]
        P = p_closed_series(pred, track, f0, f1, side)
        data = {"B": np.zeros((len(P), 2, 4)), "P": np.stack([P, P], 1), "fps": a.fps}
        an = events.signals(data)[0][:, 0]
        det = []
        for c0, c1 in events.closed_intervals(an, a.fps):
            o = events.open_onset(an, c0, c1)
            if o is not None:
                det.append(f0 + o)
        ch = st["changes"]
        human = [ch[i][0] for i in range(1, len(ch))
                 if ch[i - 1][1] == "gripper_closed" and ch[i][1] == "gripper_open"]
        video = next((v for v in segments if v.startswith(stem)), stem + ".mp4")
        seg = next((k for k, (s, e) in enumerate(segments.get(video, [])) if s <= f0 < e), None)
        split = "val" if seg is not None and seg in vseg.get(video, set()) else "train"
        used = set()
        errs = []
        for h in human:
            cand = [d for d in det if abs(d - h) <= a.win and d not in used]
            if not cand:
                miss[split] += 1
                continue
            d = min(cand, key=lambda d: abs(d - h))
            used.add(d)
            errs.append(d - h)
        extra[split] += len([d for d in det if d not in used])
        res[split] += errs
        print(f"{split:5s} {key:45s} human releases {len(human):2d}  detected {len(det):2d}  errors {errs}")

    print()
    for split in ("val", "train"):
        e = np.array(res[split], float)
        if not len(e):
            print(f"{split}: no matched releases (missed {miss[split]}, extra {extra[split]})")
            continue
        print(f"{split}: n={len(e)}  signed median {np.median(e):+.0f}f  mean {e.mean():+.1f}f  "
              f"median|e| {np.median(np.abs(e)):.0f}f  within ±25f {np.mean(np.abs(e) <= 25):.0%}  "
              f"| missed {miss[split]}  extra detections {extra[split]}")
    use = res["val"] or res["train"]
    if use:
        print(f"\n-> suggested ONSET_SHIFT = {int(round(np.median(use)))}  "
              f"(from {'val' if res['val'] else 'train (optimistic)'} units; positive = pipeline is late)")


if __name__ == "__main__":
    main()
