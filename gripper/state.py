"""Open/closed classifier on gripper crops (YOLO11-cls), fed by SAM 2 gripper tracks.

  export   crops -> <project>/dataset_cls/{train,val}/<class>/*.jpg
           sources: (1) timeline labels of DONE units (dense: every --stride-th frame, plus every frame
           within --near frames of a change point, where the classifier matters most);
           (2) every box of done items in items.json. Required videos (EVAL_VIDEOS) are never used.
           Split by segment like export_yolo: whole segments go to val.
  train    fine-tune yolo11{n,s}-cls on the crops
  predict  classify every tracked frame of a video -> <project>/pred/<stem>.json (per frame, per side
           [p_open, p_closed, p_unknown]; the label tool shows it and uses it as pre-labels),
           <stem>.sides.csv (same columns as before: state, conf, p_open, box), <stem>.p_open.png,
           and with --vis a full-res overlay video.

All crops are cut from the 1280x720 frames decoded from the MP4 (common.crop: square, box side
x 1.25, resized to 256).

  python -m gripper.state export
  python -m gripper.state train --name c1
  python -m gripper.state predict --model runs/gripper/cls/c1/weights/best.pt            # all tracked videos
  python -m gripper.state predict --model ... --videos 12_fold_four_towels.mp4 --vis
"""
import argparse
import csv
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from .common import CLASSES, DATA, PROJECT, SIDES, YOLO_DIR, Readers, Store, crop, is_eval
from .timeline import States, dense, load_all_tracks, units

COL = {"gripper_open": (0, 200, 0), "gripper_closed": (0, 0, 230), "gripper_unknown": (0, 210, 230)}


def val_segments(store, videos, frac, seed):
    rng = random.Random(seed)
    out = {}
    for v in sorted(videos):
        segs = list(range(len(store.segments.get(v, []))))
        k = max(1, round(frac * len(segs))) if len(segs) >= 2 else 0
        out[v] = set(rng.sample(segs, k))
    return out


def cmd_export(a):
    store, readers = Store(a.project), Readers(a.data)
    tracks, states = load_all_tracks(a.project), States(a.project)
    out = Path(a.out or Path(a.project) / "dataset_cls")
    videos = {t["video"] for t in tracks.values()} | {it["video"] for it in store.items}
    videos = {v for v in videos if not is_eval(v)}
    vseg = val_segments(store, videos, a.val_frac, a.seed)
    jobs = defaultdict(list)  # video -> [(frame, box, cls, split, name)]

    n_units = 0
    for key, video, ua, ub, side in units(tracks):
        st = states.get(key)
        if is_eval(video) or not st.get("done"):
            continue
        n_units += 1
        stem = Path(video).stem
        frames = tracks[stem]["frames"]
        lab = dense(st["changes"], st["bad"], ua, ub)
        near = set()
        for f, _, _ in st["changes"]:
            near.update(range(f - a.near, f + a.near + 1))
        seg = store.seg_of(video, ua)
        split = "val" if seg is not None and seg in vseg.get(video, set()) else "train"
        for f, c in lab.items():
            rec = frames.get(str(f), {}).get(side)
            if not rec or rec[5]:  # no box, suspect (size jump / L-R overlap) or filled (no mask)
                continue
            if f in near or f % a.stride == 0:
                jobs[video].append((f, rec[:4], c, split, f"{stem}_f{f:06d}_{side}"))
    n_items = 0
    for it in store.items:
        if it["status"] != "done" or is_eval(it["video"]):
            continue
        seg = it.get("seg")
        split = "val" if seg is not None and seg in vseg.get(it["video"], set()) else "train"
        for k, b in enumerate(it["boxes"]):
            if b["cls"] in CLASSES:
                jobs[it["video"]].append((it["frame"], b["xyxy"], b["cls"], split,
                                          f"{Path(it['video']).stem}_f{it['frame']:06d}_box{k}"))
                n_items += 1

    if out.exists():
        shutil.rmtree(out)
    for split in ("train", "val"):
        for c in CLASSES:
            (out / split / c).mkdir(parents=True)
    cnt = Counter()
    for video, js in jobs.items():
        r = readers(video)
        for f, box, c, split, name in sorted(js, key=lambda j: j[0]):
            img = r.get(f)
            if img is None:
                continue
            cv2.imwrite(str(out / split / c / f"{name}.jpg"), crop(img, box), [cv2.IMWRITE_JPEG_QUALITY, 95])
            cnt[(split, c)] += 1
    # every class folder must be non-empty in both splits (torchvision ImageFolder); move one image
    # train -> val if a class has none in val, drop a class that has no images at all
    for c in CLASSES:
        tr, va = sorted((out / "train" / c).glob("*.jpg")), list((out / "val" / c).glob("*.jpg"))
        if not tr and not va:
            shutil.rmtree(out / "train" / c)
            shutil.rmtree(out / "val" / c)
            print(f"WARNING: no '{c}' crops at all -> class dropped")
        elif not va:
            shutil.move(str(tr[-1]), str(out / "val" / c / tr[-1].name))
            cnt[("train", c)] -= 1
            cnt[("val", c)] += 1
            print(f"WARNING: no '{c}' crops in val -> moved one from train")
        elif not tr:
            shutil.move(str(va[-1]), str(out / "train" / c / va[-1].name))
            cnt[("val", c)] -= 1
            cnt[("train", c)] += 1
    print(f"wrote {out}  ({n_units} done timeline units, {n_items} item boxes; val segments "
          f"{ {Path(v).stem[:12]: sorted(s) for v, s in vseg.items()} })")
    for split in ("train", "val"):
        print(f"  {split:5s} " + "  ".join(f"{c}={cnt[(split, c)]}" for c in CLASSES))


def cmd_train(a):
    from ultralytics import YOLO
    data = a.out or str(Path(a.project) / "dataset_cls")
    model = YOLO(a.weights)
    model.train(data=data, imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device,
                workers=a.workers, project=f"{a.project}/cls", name=a.name, exist_ok=True,
                patience=20, cos_lr=True, seed=0,
                # keep the fingertips in the crop: mild random-resized-crop, light erasing
                scale=0.2, erasing=0.1, fliplr=0.5, flipud=0.0, hsv_h=0.015, hsv_s=0.5, hsv_v=0.35)
    m = model.val(data=data, imgsz=a.imgsz, split="val", device=a.device,
                  project=f"{a.project}/cls", name=f"{a.name}_val", exist_ok=True)
    print(f"val top1={m.top1:.3f}")
    print(f"weights: {a.project}/cls/{a.name}/weights/best.pt")


class Classifier:
    def __init__(self, weights, device=None, imgsz=224):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.idx = {n: i for i, n in self.m.names.items()}
        self.device, self.imgsz = device, imgsz
        self.name = Path(weights).parent.parent.name

    def __call__(self, crops):
        """-> array [N, 3] of [p_open, p_closed, p_unknown] (0 for a class the model lacks)."""
        res = self.m.predict(crops, imgsz=self.imgsz, device=self.device, verbose=False)
        out = np.zeros((len(crops), 3), np.float32)
        for k, r in enumerate(res):
            p = r.probs.data.float().cpu().numpy()
            for j, c in enumerate(CLASSES):
                if c in self.idx:
                    out[k, j] = p[self.idx[c]]
        return out


def p_open_of(p):
    """p = [p_open, p_closed, p_unknown] -> p_open in [0, 1], or None when the crop is 'unknown'."""
    if p[2] > max(p[0], p[1]):
        return None
    return float(p[0] / max(p[0] + p[1], 1e-6))


def predict_video(clf, readers, project, stem, t, batch=64, vis=False):
    video, frames = t["video"], t["frames"]
    r = readers(video)
    todo = sorted(int(f) for f in frames)
    pred, buf = {}, []

    def flush():
        P = clf([c for _, _, c in buf])
        for (f, side, _), p in zip(buf, P):
            pred.setdefault(str(f), {})[side] = [round(float(x), 4) for x in p]
        buf.clear()

    for f in todo:
        rec = frames[str(f)]
        if not rec:
            continue
        img = r.get(f)
        if img is None:
            break
        for side in SIDES:
            if side in rec:
                buf.append((f, side, crop(img, rec[side][:4])))
        if len(buf) >= batch:
            flush()
    if buf:
        flush()
    outd = Path(project) / "pred"
    outd.mkdir(parents=True, exist_ok=True)
    (outd / f"{stem}.json").write_text(json.dumps({"video": video, "model": clf.name, "frames": pred}))

    fps = t["fps"]
    with open(outd / f"{stem}.sides.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "t"] + [f"{s}_{k}" for s in SIDES
                                     for k in ("state", "conf", "p_open", "x1", "y1", "x2", "y2")])
        for f in todo:
            row = [f, f"{f / fps:.3f}"]
            for side in SIDES:
                rec, p = frames[str(f)].get(side), pred.get(str(f), {}).get(side)
                if rec and p:
                    k = int(np.argmax(p))
                    po = p_open_of(p)
                    row += [CLASSES[k], f"{p[k]:.4f}", "" if po is None else f"{po:.4f}", *rec[:4]]
                else:
                    row += [""] * 7
            w.writerow(row)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(16, 5), sharex=True)
        for axi, side in zip(ax, SIDES):
            pts = [(f / fps, p_open_of(pred[str(f)][side])) for f in todo
                   if side in pred.get(str(f), {}) and p_open_of(pred[str(f)][side]) is not None]
            if pts:
                axi.plot(*zip(*pts), ".", ms=1.5)
            axi.set_ylabel(f"{side} p_open"); axi.set_ylim(-0.05, 1.05); axi.grid(alpha=0.3)
        ax[-1].set_xlabel("t (s)"); fig.suptitle(f"{stem} ({clf.name})"); fig.tight_layout()
        fig.savefig(outd / f"{stem}.p_open.png", dpi=110); plt.close(fig)
    except ImportError:
        pass
    if vis:
        vw = cv2.VideoWriter(str(outd / f"{stem}.vis.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (r.w, r.h))
        for f, img in r:
            for side in SIDES:
                rec, p = frames.get(str(f), {}).get(side), pred.get(str(f), {}).get(side)
                if rec:
                    x1, y1, x2, y2 = map(int, rec[:4])
                    c = CLASSES[int(np.argmax(p))] if p else "gripper_unknown"
                    po = p_open_of(p) if p else None
                    cv2.rectangle(img, (x1, y1), (x2, y2), COL[c], 2)
                    cv2.putText(img, f"{side} {c[8:]}" + ("" if po is None else f" p={po:.2f}"),
                                (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL[c], 2)
            cv2.putText(img, f"f={f} t={f / fps:.2f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            vw.write(img)
        vw.release()
    n = sum(len(v) for v in pred.values())
    print(f"{stem}: {n} gripper crops classified -> {outd}/{stem}.json, .sides.csv, .p_open.png")


def cmd_predict(a):
    readers = Readers(a.data)
    clf = Classifier(a.model, a.device, a.imgsz)
    tracks = load_all_tracks(a.project)
    stems = [Path(v).stem for v in a.videos] if a.videos else sorted(tracks)
    for stem in stems:
        if stem not in tracks:
            print(f"{stem}: no tracks (run python -m gripper.track first)")
            continue
        predict_video(clf, readers, a.project, stem, tracks[stem], vis=a.vis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["export", "train", "predict"])
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--out", default=None, help="dataset dir (default <project>/dataset_cls)")
    ap.add_argument("--stride", type=int, default=5, help="export: every n-th labelled frame ...")
    ap.add_argument("--near", type=int, default=10, help="export: ... plus every frame this close to a change")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights", default=f"{YOLO_DIR}/yolo11s-cls.pt")
    ap.add_argument("--name", default="c1")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--imgsz", type=int, default=224)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default=None)
    ap.add_argument("--model", default=None, help="predict: classifier weights")
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--vis", action="store_true")
    a = ap.parse_args()
    {"export": cmd_export, "train": cmd_train, "predict": cmd_predict}[a.cmd](a)


if __name__ == "__main__":
    main()
