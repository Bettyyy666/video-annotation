"""Choose frames to label and pre-label them.

  segments  pick random, non-overlapping segments of the training videos (default: TRAIN_VIDEOS in
            common.py = 09 cables, 14 wipe-down, 21 dry goods) and stratified-random frames inside
            each segment (round 0). Run again to add more segments.
  mine      run a trained model over the segments (full-res, every --stride frames) and add frames
            where it is unsure: open<->closed flips (the switch lies between two samples) and frames
            with p_open near 0.5 / unknown / a missing gripper. Model boxes become pre-labels.
  prelabel  (re)fill model boxes on items not yet reviewed by a human
  status    counts
  anchor    add frames of a video as anchor items for the SAM 2 tracker (gripper.track), e.g. frame 0
            of a required video; draw / fix the two gripper boxes in the label tool

Examples (run from video-annotation/):
  python -m gripper.prepare segments                       # 3 training videos, 6 x 10 s each
  python -m gripper.prepare prelabel --model yoloe:$YOLO/yoloe/yoloe-11l-seg.pt
  python -m gripper.prepare mine --model runs/gripper/train/r1/weights/best.pt
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

import cv2

from .common import CLASSES, DATA, PENDING, PROJECT, TRAIN_VIDEOS, Readers, Store, YOLO_DIR
from .tracks import assign_sides, p_open


def _free(f, taken, gap):
    return all(abs(f - g) >= gap for g in taken)


def _place_segments(n_frames, n_seg, length, existing, rng):
    """n_seg random [s, e) segments of `length` frames, not overlapping `existing` (with a gap of
    length/2 so neighbouring segments are not near-duplicates). The first call spreads them out:
    one segment at a random position inside each of n_seg equal parts of the video."""
    gap = length // 2
    ok = lambda s: all(s + length + gap <= a or s >= b + gap for a, b in existing + out)
    out = []
    if not existing:
        part = n_frames / n_seg
        for k in range(n_seg):
            lo, hi = int(k * part), int((k + 1) * part) - length
            for _ in range(200):
                s = rng.randint(lo, max(lo, hi))
                if s + length <= n_frames and ok(s):
                    out.append([s, s + length])
                    break
    for _ in range(5000):  # top up (or add segments on a later call) anywhere in the video
        if len(out) >= n_seg:
            break
        s = rng.randint(0, n_frames - length)
        if ok(s):
            out.append([s, s + length])
    return sorted(out)


def cmd_segments(a, store, readers):
    rng = random.Random(a.seed + len(store.segments))
    for v in a.videos:
        r = readers(v)
        n_frames = r.n_meta - 1  # last metadata frame is sometimes not decodable
        length = int(round(a.seg_sec * r.fps))
        existing = store.segments.setdefault(v, [])
        new = _place_segments(n_frames, a.n_seg, length, list(existing), rng)
        taken = [it["frame"] for it in store.items if it["video"] == v]
        added = 0
        for s, e in new:
            existing.append([s, e])
            k = len(existing) - 1
            edges = [s + round(j * (e - s) / a.per_seg) for j in range(a.per_seg + 1)]
            for lo, hi in zip(edges[:-1], edges[1:]):
                for _ in range(20):
                    f = rng.randrange(lo, max(lo + 1, hi))
                    if _free(f, taken, a.min_gap):
                        store.add(v, f, r.fps, "segment", seg=k)
                        taken.append(f)
                        added += 1
                        break
        segs = ", ".join(f"{s / r.fps:.1f}-{e / r.fps:.1f}s" for s, e in new)
        print(f"{v}: +{len(new)} segments [{segs}], +{added} frames "
              f"({r.w}x{r.h}, {n_frames} frames, {r.fps:g} fps)")


def _run_ranges(det, reader, ranges, stride, batch=16):
    """Run the detector every `stride` frames inside each [s, e) range (frame-exact reads)."""
    ids, imgs, per = [], [], []
    for s, e in ranges:
        for f in range(s, e, stride):
            img = reader.get(f)
            if img is None:
                break
            ids.append(f)
            imgs.append(img)
            if len(imgs) == batch:
                per += det(imgs)
                imgs = []
    if imgs:
        per += det(imgs)
    return ids, per


def cmd_mine(a, store, readers):
    from .models import Detector
    det = Detector(a.model, device=a.device, imgsz=a.imgsz)
    rng = random.Random(a.seed)
    videos = a.videos or sorted(store.segments)
    for v in videos:
        r = readers(v)
        ranges = [[0, r.n_meta - 1]] if a.whole_video else store.segments.get(v, [])
        if not ranges:
            print(f"{v}: no segments (run `segments` first or pass --whole-video)")
            continue
        ids, per = _run_ranges(det, r, ranges, a.stride)
        sides = assign_sides(per, ids, r.w)
        cache = store.dir / "mine" / f"{Path(v).stem}__{Path(a.model).stem}.json"
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps({"frames": ids, "instances": per}))

        flips, unsure = [], []
        for side in ("L", "R"):
            prev = None  # (frame, state); flips only between nearby samples, never across segments
            for f, s in zip(ids, sides):
                p = p_open(s[side])
                if p is None:
                    unsure.append((0.5, f, f"{side}:missing_or_unknown"))
                    continue
                unsure.append((1 - abs(2 * p - 1), f, f"{side}:p_open={p:.2f}"))
                st = p >= 0.5
                if prev and prev[1] != st and f - prev[0] <= 3 * a.stride:
                    flips.append((prev[0], f, f"{side}:{'close' if prev[1] else 'open'}"))
                prev = (f, st)

        taken = [it["frame"] for it in store.items if it["video"] == v]
        n_flip = n_uns = 0
        rng.shuffle(flips)
        for lo, hi, why in flips:
            if n_flip >= a.k_flip:
                break
            for f in sorted({(lo + hi) // 2, rng.randint(lo + 1, hi)}):
                if _free(f, taken, a.min_gap // 2):
                    store.add(v, f, r.fps, f"mined:flip:{why}")
                    taken.append(f)
                    n_flip += 1
        for u, f, why in sorted(unsure, reverse=True):
            if n_uns >= a.k_uncertain or u < 0.3:
                break
            if _free(f, taken, a.min_gap):
                store.add(v, f, r.fps, f"mined:unsure:{why}")
                taken.append(f)
                n_uns += 1
        print(f"{v}: {len(ids)} frames checked, {len(flips)} flips, "
              f"+{n_flip} flip frames, +{n_uns} uncertain frames")
    a.overwrite = False
    cmd_prelabel(a, store, readers, det)  # pre-label the new frames with the same model


def cmd_prelabel(a, store, readers, det=None):
    from .models import Detector
    det = det or Detector(a.model, device=a.device, imgsz=a.imgsz, conf=0.02)
    # YOLOE "robot gripper" scores are low at 1280 even for correct, tight boxes
    min_conf = a.min_conf if a.min_conf is not None else (0.03 if det.name.startswith("yoloe") else 0.10)
    store.ensure_frames(readers)
    todo = [it for it in store.items if it["status"] == "todo" and it.get("source") != "anchor"
            and not any(b["by"] == "human" for b in it["boxes"])
            and (a.overwrite or not it["boxes"])]
    for k in range(0, len(todo), 16):
        chunk = todo[k:k + 16]
        imgs = [cv2.imread(str(store.frame_path(it))) for it in chunk]  # lossless copy of the frame
        for it, insts in zip(chunk, det(imgs)):
            it["boxes"] = [{"cls": i["cls"], "xyxy": i["xyxy"], "by": det.name, "conf": i["conf"]}
                           for i in insts if i["conf"] >= min_conf]
    store.save()
    print(f"pre-labelled {len(todo)} items with {det.name}")


def cmd_status(a, store, readers):
    st = Counter((it["video"], it["status"]) for it in store.items)
    for v in sorted({it["video"] for it in store.items}):
        segs = store.segments.get(v, [])
        print(f"{v:42s} segments={len(segs):2d}  " +
              "  ".join(f"{s}={st[(v, s)]}" for s in ("todo", "done", "skip")))
    boxes = Counter(b["cls"] for it in store.items if it["status"] == "done" for b in it["boxes"])
    print("done boxes:", {c: boxes[c] for c in CLASSES})
    pend = sum(b["cls"] == PENDING for it in store.items for b in it["boxes"])
    print(f"'{PENDING}' boxes still waiting for a state: {pend}")


def cmd_anchor(a, store, readers):
    """Add frames of one video as anchor items (boxes for the SAM 2 tracker). If the video is already
    tracked, the tracked boxes are copied in as blue 'gripper' boxes to correct; otherwise draw them."""
    from .track import load_tracks
    if not a.video or not a.frames:
        raise SystemExit("anchor: pass --video and --frames")
    r = readers(a.video)
    tr = load_tracks(store.dir, Path(a.video).stem) or {"frames": {}}
    for f in a.frames:
        rec = tr["frames"].get(str(f), {})
        boxes = [{"cls": PENDING, "xyxy": v[:4], "by": "track"} for v in rec.values()]
        it = store.add(a.video, f, r.fps, "anchor", boxes=boxes)
        print(f"{a.video} frame {f}: " + ("added" if it else "already an item") +
              (f" with {len(boxes)} tracked boxes" if it and boxes else ""))
    store.ensure_frames(readers)
    store.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["segments", "mine", "prelabel", "status", "anchor"])
    ap.add_argument("--video", default=None, help="anchor: video file name")
    ap.add_argument("--frames", type=int, nargs="*", default=None, help="anchor: frame indices")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--videos", nargs="*", default=None,
                    help="file names inside --data (segments: default TRAIN_VIDEOS; "
                         "mine: default all videos that have segments)")
    ap.add_argument("--n-seg", type=int, default=6, help="segments: new segments per video")
    ap.add_argument("--seg-sec", type=float, default=10.0, help="segments: segment length (s)")
    ap.add_argument("--per-seg", type=int, default=8, help="segments: frames per segment")
    ap.add_argument("--min-gap", type=int, default=25, help="min frames between picked frames")
    ap.add_argument("--model", default=f"yoloe:{YOLO_DIR}/yoloe/yoloe-11l-seg.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-conf", type=float, default=None,
                    help="pre-label box threshold (default 0.03 for YOLOE, 0.10 for a trained model)")
    ap.add_argument("--overwrite", action="store_true", help="prelabel: replace existing model boxes")
    ap.add_argument("--stride", type=int, default=5, help="mine: run the model every N frames")
    ap.add_argument("--whole-video", action="store_true", help="mine: whole video, not only segments")
    ap.add_argument("--k-flip", type=int, default=12, help="mine: max flip frames per video")
    ap.add_argument("--k-uncertain", type=int, default=8, help="mine: max uncertain frames per video")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    store, readers = Store(a.project), Readers(a.data)
    if a.cmd == "segments":
        a.videos = a.videos or TRAIN_VIDEOS
        cmd_segments(a, store, readers)
        store.ensure_frames(readers)
        store.save()
    elif a.cmd == "mine":
        cmd_mine(a, store, readers)
    elif a.cmd == "anchor":
        cmd_anchor(a, store, readers)
    elif a.cmd == "prelabel":
        cmd_prelabel(a, store, readers)
    else:
        cmd_status(a, store, readers)


if __name__ == "__main__":
    main()
