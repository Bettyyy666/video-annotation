"""items.json (status == done) -> Ultralytics YOLO dataset.

Adjacent frames are near-duplicates, so a random frame split would leak. Default split is BY SEGMENT:
in every training video ~val_frac of its random segments (at least one) go to val, the rest to
train, so all three videos (and their robots / scenes) are seen in training and validation.
Segments are at least half a segment length apart, so frames across the split are not neighbours.
Items outside any segment (mine --whole-video) go to train.
`--split video` holds out whole videos instead.

Images are symlinks to the lossless full-res PNGs (1280x720), labels are YOLO txt.

  python -m gripper.export_yolo                         # split by segment
  python -m gripper.export_yolo --split video --val-videos 21_portion_dry_goods_into_jars.mp4
"""
import argparse
import os
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import cv2

from .common import CLASSES, CLS_ID, PENDING, PROJECT, Store, is_eval


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--out", default=None, help="default <project>/dataset")
    ap.add_argument("--split", choices=["segment", "video"], default="segment")
    ap.add_argument("--val-videos", nargs="*", default=None, help="--split video: explicit list")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--copy", action="store_true", help="copy PNGs instead of symlinking")
    a = ap.parse_args()

    store = Store(a.project)
    out = Path(a.out or store.dir / "dataset")
    done = [it for it in store.items if it["status"] == "done" and not is_eval(it["video"])]
    if not done:
        raise SystemExit("no items with status 'done'")
    bad = [it["id"] for it in done if any(b["cls"] == PENDING for b in it["boxes"])]
    if bad:
        raise SystemExit(f"done items still have '{PENDING}' boxes without a state: {bad[:10]}")

    by_video = defaultdict(list)
    for it in done:
        by_video[it["video"]].append(it)

    if a.split == "segment":
        rng = random.Random(a.seed)
        val_segs = {}
        for v in sorted(by_video):
            segs = sorted({it.get("seg") for it in by_video[v] if it.get("seg") is not None})
            k = max(1, round(a.val_frac * len(segs))) if len(segs) >= 2 else 0
            val_segs[v] = set(rng.sample(segs, k))
        is_val = lambda it: it.get("seg") is not None and it["seg"] in val_segs[it["video"]]
        desc = {v: sorted(s) for v, s in val_segs.items()}
        if not any(val_segs.values()):
            print("WARNING: need >= 2 labelled segments in a video for a val split; val = train")
    else:
        if a.val_videos is None:  # smallest videos first until ~val_frac of the frames
            val, n = [], 0
            for v in sorted(by_video, key=lambda v: len(by_video[v])):
                if n >= a.val_frac * len(done) or len(val) >= len(by_video) - 1:
                    break
                val.append(v)
                n += len(by_video[v])
            a.val_videos = val
        is_val = lambda it: it["video"] in a.val_videos
        desc = a.val_videos
        if not a.val_videos:
            print("WARNING: need >= 2 labelled videos for a val split; val = train")

    if out.exists():
        shutil.rmtree(out)
    stats = Counter()
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True)
        (out / "labels" / split).mkdir(parents=True)
    for it in done:
        split = "val" if is_val(it) else "train"
        src = store.frame_path(it).resolve()
        h, w = cv2.imread(str(src)).shape[:2]  # 720, 1280
        name = it["id"].replace("/", "__")
        dst = out / "images" / split / f"{name}.png"
        shutil.copy(src, dst) if a.copy else os.symlink(src, dst)
        lines = []
        for b in it["boxes"]:
            x1, y1, x2, y2 = b["xyxy"]
            lines.append(f"{CLS_ID[b['cls']]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                         f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            stats[(split, b["cls"])] += 1
        (out / "labels" / split / f"{name}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        stats[(split, "images")] += 1

    val_dir = "images/val" if stats[("val", "images")] else "images/train"
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: {val_dir}\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)))
    print(f"wrote {out}/data.yaml   split={a.split}   val: {desc}")
    for split in ("train", "val"):
        print(f"  {split:5s} images={stats[(split, 'images')]:4d}  " +
              "  ".join(f"{c}={stats[(split, c)]}" for c in CLASSES))


if __name__ == "__main__":
    main()
