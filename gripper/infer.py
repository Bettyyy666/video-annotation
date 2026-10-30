"""Run the trained gripper detector on every frame of a video, decoded at full resolution.

Outputs in --out (default <project>/infer/<model-name>/):
  <stem>.det.jsonl   one line per frame: {"f", "t", "inst": [...all grouped instances...]}
  <stem>.sides.csv   per frame, LEFT and RIGHT gripper: state, conf, p_open (empty = missing), box
  <stem>.p_open.png  the two p_open(t) curves (quick visual check)
  <stem>.vis.mp4     (--vis) full-res overlay video

Missing values are left missing here; smoothing / HMM / bias correction happen downstream.

  python -m gripper.infer --model runs/gripper/train/r1/weights/best.pt --videos 12_fold_four_towels.mp4
"""
import argparse
import csv
import json
from pathlib import Path

import cv2

from .common import DATA, PROJECT, FrameReader
from .models import Detector
from .tracks import assign_sides, p_open

COL = {"gripper_open": (0, 200, 0), "gripper_closed": (0, 0, 230), "gripper_unknown": (0, 210, 230)}


def run(det, path, out, batch=16, vis=False, side_conf=0.25):
    r = FrameReader(path)
    stem = Path(path).stem
    out.mkdir(parents=True, exist_ok=True)
    ids, per, buf = [], [], []
    for f, img in r:  # sequential decode: every frame, exact, 1280x720
        buf.append((f, img))
        if len(buf) == batch:
            per += det([b[1] for b in buf]); ids += [b[0] for b in buf]; buf = []
        if f % 1000 == 0:
            print(f"  {stem}: frame {f}/{r.n_meta}", flush=True)
    if buf:
        per += det([b[1] for b in buf]); ids += [b[0] for b in buf]
    sides = assign_sides(per, ids, r.w, conf_min=side_conf)

    with open(out / f"{stem}.det.jsonl", "w") as fh:
        for f, insts in zip(ids, per):
            fh.write(json.dumps({"f": f, "t": round(f / r.fps, 3), "inst": insts}) + "\n")
    with open(out / f"{stem}.sides.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t"] + [f"{s}_{k}" for s in ("L", "R")
                                     for k in ("state", "conf", "p_open", "x1", "y1", "x2", "y2")])
        for f, s in zip(ids, sides):
            row = [f, f"{f / r.fps:.3f}"]
            for side in ("L", "R"):
                i = s[side]
                p = p_open(i)
                row += ([i["cls"], i["conf"], "" if p is None else f"{p:.4f}", *i["xyxy"]] if i
                        else ["", "", "", "", "", "", ""])
            w.writerow(row)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
        for a, side in zip(ax, ("L", "R")):
            t = [f / r.fps for f, s in zip(ids, sides) if p_open(s[side]) is not None]
            p = [p_open(s[side]) for s in sides if p_open(s[side]) is not None]
            a.plot(t, p, ".", ms=1.5)
            a.set_ylabel(f"{side} p_open"); a.set_ylim(-0.05, 1.05); a.grid(alpha=0.3)
        ax[-1].set_xlabel("t (s)"); fig.suptitle(f"{stem}  ({det.name})"); fig.tight_layout()
        fig.savefig(out / f"{stem}.p_open.png", dpi=110); plt.close(fig)
    except ImportError:
        pass

    if vis:
        vw = cv2.VideoWriter(str(out / f"{stem}.vis.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), r.fps, (r.w, r.h))
        for (f, img), s in zip(r, sides):
            for side in ("L", "R"):
                i = s[side]
                if i:
                    x1, y1, x2, y2 = map(int, i["xyxy"])
                    cv2.rectangle(img, (x1, y1), (x2, y2), COL[i["cls"]], 2)
                    p = p_open(i)
                    cv2.putText(img, f"{side} {i['cls'][8:]} {'' if p is None else f'p={p:.2f}'}",
                                (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[i["cls"]], 2)
            cv2.putText(img, f"f={f} t={f / r.fps:.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            vw.write(img)
        vw.release()
    print(f"{stem}: {len(ids)} frames -> {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.05, help="raw box threshold (keep low)")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None)
    ap.add_argument("--vis", action="store_true")
    a = ap.parse_args()
    det = Detector(a.model, device=a.device, imgsz=a.imgsz, conf=a.conf)
    out = Path(a.out or f"{a.project}/infer/{det.name.replace('/', '_')}")
    for v in a.videos:
        run(det, Path(a.data) / v, out, a.batch, a.vis)


if __name__ == "__main__":
    main()
