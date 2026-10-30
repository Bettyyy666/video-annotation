"""Track the LEFT and RIGHT gripper through frame ranges with SAM 2,
starting from human anchor boxes.

Why: a detector has to find the gripper again in every frame and fails on scenes it has not seen
(video 12: no box in ~70 % of frames). SAM 2 only needs one box per gripper and then follows it,
so it works in any scene and keeps the left/right identity by construction.

How (per range [a, b], e.g. a 10 s training segment or a whole required video):
  * anchors = frames with human boxes (done items in items.json, see `pick_anchors`);
  * from the first anchor, run SAM 2 forward in windows of --window frames, and backward to `a`;
  * each window is prompted per gripper with the anchor box, or the box at the end of the previous
    window, turned into a mask by SAM 2's image predictor (of its 3 candidates the one whose box fits
    the prompt box best; a plain box prompt sometimes picked one finger or the held object);
  * at a later anchor the tracker is re-prompted with the human boxes (this is how you correct it);
  * every frame gets a box: SAM 2's mask box whatever its size; when the mask vanishes the last box
    is carried over (marked 'filled'); a sudden size jump between consecutive frames is kept but
    marked suspect. The start of every filled / suspect run is listed in "flags", so you know where
    to look and, if needed, add an anchor in the label tool (timeline page, key A).

Frames for SAM 2 are decoded from the MP4 at 1280x720 and written as quality-95 JPEGs (SAM 2's
video API only reads JPEG folders); boxes are returned in original pixel coordinates.

Output: <project>/tracks/<stem>.json
  {"video", "fps", "w", "h", "ranges": [[a, b], ...], "anchors": {f: {"L": box, "R": box}},
   "frames": {f: {"L": [x1, y1, x2, y2, score, mark], "R": ...}}   mark: 0 ok, 1 suspect, 2 filled, "flags": [[f, side, why], ...],
   "check": [...IoU of the track against human boxes that were NOT used as anchors...]}

  python -m gripper.track segments                 # all training segments that have a done item
  python -m gripper.track video --videos 12_fold_four_towels.mp4 26_sort_cutlery_drawer_insert.mp4
  python -m gripper.track report                   # why is the check IoU low? (images, no GPU)
"""
import argparse
import json
import os

os.environ.setdefault("TQDM_DISABLE", "1")  # SAM 2 prints two progress bars per window
import shutil
import time
from pathlib import Path

import cv2
import numpy as np

from .common import DATA, PENDING, PROJECT, SAM2_MODELS, SIDES, Readers, Store, is_eval

OBJ = {"L": 1, "R": 2}
SIDE_OF = {1: "L", 2: "R"}
MIN_AREA = 150          # px at 1280x720; smaller masks = gripper not visible
JUMP = 2.5              # box area change between consecutive frames beyond this = suspect (kept, flagged)
SIZE_TOL = 3.0          # mask box area outside [1/3, 3] x the human anchor box: keep the centre, use the last
                        # plausible size (marked suspect); prevents a finger / the arm from taking over the track


def box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / u if u > 0 else 0.0


def area(b):
    return max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])


def mask_box(m):
    """Box of the largest connected component (SAM 2 masks sometimes have small far-away specks)."""
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 1:
        return None, 0
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    x, y, w, h, a = st[k]
    return [float(x), float(y), float(x + w), float(y + h)], int(a)


def assign_sides(boxes, pred=None, width=1280):
    """Human boxes of one anchor frame -> {side: box}. Match to the current track if there is one,
    otherwise by x position (left half / left of the other box = L)."""
    boxes = [list(map(float, b)) for b in boxes][:2]
    if not boxes:
        return {}
    if pred and all(pred.get(s) for s in SIDES):
        if len(boxes) == 1:
            s = max(SIDES, key=lambda s: box_iou(boxes[0], pred[s]))
            return {s: boxes[0]}
        keep = box_iou(boxes[0], pred["L"]) + box_iou(boxes[1], pred["R"])
        swap = box_iou(boxes[0], pred["R"]) + box_iou(boxes[1], pred["L"])
        return {"L": boxes[0], "R": boxes[1]} if keep >= swap else {"L": boxes[1], "R": boxes[0]}
    if len(boxes) == 1:
        cx = (boxes[0][0] + boxes[0][2]) / 2
        return {"L" if cx < width / 2 else "R": boxes[0]}
    boxes.sort(key=lambda b: b[0] + b[2])
    return {"L": boxes[0], "R": boxes[1]}


def pick_anchors(items, video, ranges, all_anchors=False):
    """Done items of `video` with real boxes. Anchors: items added as anchors, plus the earliest done
    item of each range (unless --all-anchors). The rest are kept to measure tracking quality."""
    done = sorted((it for it in items if it["video"] == video and it["status"] == "done"
                   and any(b["cls"] != PENDING for b in it["boxes"])), key=lambda it: it["frame"])
    anchors, check = [], []
    for a, b in ranges:
        inr = [it for it in done if a <= it["frame"] <= b]
        for k, it in enumerate(inr):
            (anchors if all_anchors or k == 0 or it.get("source") == "anchor" else check).append(it)
    return anchors, check


class SamTracker:
    def __init__(self, model="base_plus", device=None, ckpt=None, window=50, tmp=None):
        import torch
        from sam2.build_sam import build_sam2_video_predictor
        self.torch = torch
        if device in (None, "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = f"cuda:{device}" if str(device).isdigit() else device
        ck, cfg = SAM2_MODELS[model]
        ck = ckpt or ck
        if not os.path.exists(ck):
            raise SystemExit(f"SAM 2 checkpoint not found: {ck} (pass --ckpt, see README.md section 1)")
        self.vp = build_sam2_video_predictor(cfg, ck, device=self.dev)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        self.ip = SAM2ImagePredictor(build_sam2(cfg, ck, device=self.dev))
        self.window = window
        self.tmp = Path(tmp or "/tmp") / f"gripper_sam2_{os.getpid()}"
        print(f"[SamTracker] sam2={model} device={self.dev} window={window}", flush=True)

    def _prompt_mask(self, img, box):
        """Box -> mask for prompting the video predictor. SAM 2 returns 3 candidate masks for a box (a part,
        the object, a larger whole); a box prompt straight into the video predictor sometimes takes the part
        (one finger, the held fork: 26_sort_cutlery anchor at frame 6600 came out at 0.13x the box). Take the
        candidate whose own box agrees best with the prompt box. -> (mask, IoU of its box with the prompt)."""
        torch = self.torch
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if "cuda" in self.dev else torch.no_grad()
        with torch.inference_mode(), ctx:
            self.ip.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            masks, scores, _ = self.ip.predict(box=np.asarray(box[:4], np.float32), multimask_output=True)
        best, bi = None, -1.0
        for m, sc in zip(masks, scores):
            mb, a = mask_box(m > 0.5)
            if mb is None:
                continue
            v = box_iou(mb, box) + 0.05 * float(sc)
            if v > bi:
                best, bi = m > 0.5, v
        return best, bi

    # ---- one SAM 2 window
    def _window(self, reader, frames, prompts, reverse):
        """frames: ascending global indices; prompts {side: box} at frames[-1] if reverse else frames[0].
        -> {f: {side: [x1, y1, x2, y2, score]}} (side missing = not visible)."""
        torch = self.torch
        shutil.rmtree(self.tmp, ignore_errors=True)
        self.tmp.mkdir(parents=True)
        n, imgs = 0, {}
        for k, f in enumerate(frames):
            img = reader.get(f)
            if img is None:
                break
            cv2.imwrite(str(self.tmp / f"{k:05d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            imgs[k] = img
            n += 1
        if n < len(frames):  # end of the decodable video
            frames = frames[:n]
            if reverse or not frames:
                return {}
        li0 = len(frames) - 1 if reverse else 0
        out = {}
        ctx = torch.autocast("cuda", dtype=torch.bfloat16) if "cuda" in self.dev else torch.no_grad()
        with torch.inference_mode(), ctx:
            state = self.vp.init_state(video_path=str(self.tmp), offload_video_to_cpu=True)
            self.prompt_fit = {}
            for side, box in prompts.items():
                m, fit = self._prompt_mask(imgs[li0], box)
                self.prompt_fit[side] = fit
                if m is not None and fit >= 0.3:
                    self.vp.add_new_mask(state, frame_idx=li0, obj_id=OBJ[side], mask=m)
                else:  # no candidate matches the box: fall back to the plain box prompt
                    self.vp.add_new_points_or_box(state, frame_idx=li0, obj_id=OBJ[side],
                                                  box=np.asarray(box[:4], np.float32))
            for li, ids, logits in self.vp.propagate_in_video(state, start_frame_idx=li0, reverse=reverse):
                rec = {}
                for k, oid in enumerate(ids):
                    lg = logits[k, 0].float()
                    sel = lg > 0
                    if int(sel.sum()) < MIN_AREA:
                        continue
                    b, a = mask_box(sel.cpu().numpy())
                    if b is None or a < MIN_AREA:
                        continue
                    score = float(torch.sigmoid(lg[sel]).mean())
                    rec[SIDE_OF[int(oid)]] = [round(v, 1) for v in b] + [round(score, 3)]
                out[frames[li]] = rec
            self.vp.reset_state(state)
        return out

    # ---- one direction from a start anchor
    def _pass(self, reader, start, stop, anchors, res, flags, W):
        """Track from anchor `start` towards `stop` (either direction) in windows of W frames.

        EVERY frame gets a box for every side the anchor has:
          * SAM 2's mask box whenever there is a mask, whatever its size (the gripper can come right
            up to the camera and be several times larger than on the anchor frame);
          * if the mask vanishes (gripper fully hidden / out of view), the last box is carried over
            and marked 'filled' (rec[5] = 2);
          * a sudden size jump between consecutive frames (> JUMP x) is kept but marked suspect
            (rec[5] = 1) and listed in flags - a likely place to add an anchor;
          * a mask box > SIZE_TOL x or < 1/SIZE_TOL x the human anchor box (the mask became the arm, or
            one finger / the held fork) keeps its centre but gets the last plausible size (suspect).
        Each window is prompted with a MASK: SAM 2's 3 candidates for the prompt box, the one whose box
        agrees best with the prompt box (see _prompt_mask).
        Suspect / filled frames are shown in the label tool and left out of the classifier dataset.
        The next window is prompted with the box of the window's last frame."""
        step = 1 if stop >= start else -1
        prompts = dict(anchors[start])
        last_box = {k: list(b[:4]) for k, b in prompts.items()}
        human = {k: area(b) for k, b in prompts.items()}
        plaus = {k: list(b[:4]) for k, b in prompts.items()}   # last box of a plausible size
        check_anchor = set(prompts)
        run = {k: None for k in SIDES}  # current flagged run per side ('filled' / 'jump' / 'size')
        s = start
        while s != stop:
            nxt = [f for f in anchors if (f - s) * step > 0 and (stop - f) * step >= 0]
            nxt = (min if step > 0 else max)(nxt) if nxt else None
            e = s + step * W
            e = min(e, stop) if step > 0 else max(e, stop)
            if nxt is not None and (e - nxt) * step > 0:
                e = nxt
            if not prompts:  # the anchor had no box at all: nothing to track until the next anchor
                for f in range(s, e + step, step):
                    res.setdefault(f, {})
                if nxt is None:
                    break
                s, prompts = nxt, dict(anchors[nxt])
                last_box.update({k: list(b[:4]) for k, b in prompts.items()})
                plaus.update({k: list(b[:4]) for k, b in prompts.items()})
                human.update({k: area(b) for k, b in prompts.items()})
                check_anchor |= set(prompts)
                continue
            frames = list(range(min(s, e), max(s, e) + 1))
            out = self._window(reader, frames, prompts, reverse=step < 0)
            if not out:
                break
            order = frames if step > 0 else frames[::-1]
            first = out.get(order[0], {})
            for side in list(check_anchor):  # does SAM 2's mask on the anchor frame match the human box?
                if side in first:
                    hr = area(first[side]) / max(1.0, human.get(side, 1.0))
                    if hr > 3 or hr < 0.15:
                        flags.append([order[0], side, f"anchor mask is x{hr:.2f} the human box -> check this anchor"])
                    check_anchor.discard(side)
            for f in order:
                if f not in out:
                    continue
                rec = {}
                for side in prompts:
                    v = out[f].get(side)
                    if v is None:
                        rec[side] = last_box[side] + [0.0, 2]
                        why = "no mask -> last box carried over"
                        kind = "filled"
                    else:
                        hr = area(v) / max(1.0, human.get(side, area(v)))
                        if hr > SIZE_TOL or hr < 1 / SIZE_TOL:
                            # the mask grew onto the arm / shrank to a finger or the held object: keep its
                            # position (centre) but the size of the last plausible box, and prompt with that
                            cx, cy = (v[0] + v[2]) / 2, (v[1] + v[3]) / 2
                            pw, ph = plaus[side][2] - plaus[side][0], plaus[side][3] - plaus[side][1]
                            nb = [round(cx - pw / 2, 1), round(cy - ph / 2, 1), round(cx + pw / 2, 1),
                                  round(cy + ph / 2, 1)]
                            rec[side] = nb + [v[4], 1]
                            last_box[side] = nb
                            why = f"mask box x{hr:.2f} of the anchor box -> resized to the last plausible size"
                            kind = "size"
                        else:
                            r = area(v) / max(1.0, area(last_box[side]))
                            jump = r > JUMP or r < 1 / JUMP
                            rec[side] = v + [1 if jump else 0]
                            last_box[side] = v[:4]
                            plaus[side] = v[:4]
                            why = f"box size x{r:.2f} from one frame to the next"
                            kind = "jump" if jump else None
                    if kind and run[side] != kind:
                        flags.append([f, side, why])
                    run[side] = kind
                if "L" in rec and "R" in rec and box_iou(rec["L"], rec["R"]) > 0.5:
                    for side in SIDES:
                        rec[side][5] = max(rec[side][5], 1)
                    flags.append([f, "LR", "left and right boxes overlap (one may have jumped)"])
                res[f] = rec
            last = order[-1]
            prompts = {side: list(last_box[side]) for side in prompts}
            if last in anchors:  # re-prompt with the human boxes, matched to the current track
                anchors[last] = assign_sides(list(anchors[last].values()), res.get(last) or None, reader.w)
                prompts.update(anchors[last])
                last_box.update({k: list(b[:4]) for k, b in anchors[last].items()})
                plaus.update({k: list(b[:4]) for k, b in anchors[last].items()})
                human.update({k: area(b) for k, b in anchors[last].items()})
                check_anchor |= set(anchors[last])
                res[last] = {k: list(b[:4]) + [1.0, 0] for k, b in anchors[last].items()}
            if last == stop:
                break
            s = last

    def track_range(self, reader, a, b, anchor_boxes):
        """anchor_boxes {f: [box, ...]} (human boxes, any order). -> res {f: {side: rec}}, flags"""
        res, flags = {}, []
        fs = sorted(f for f in anchor_boxes if a <= f <= b)
        if not fs:
            return res, flags
        anchors = {f: assign_sides(anchor_boxes[f], None, reader.w) for f in fs}
        t0 = time.time()
        self._pass(reader, fs[0], b, anchors, res, flags, self.window)
        if fs[0] > a:
            self._pass(reader, fs[0], a, anchors, res, flags, self.window)
        for f in fs:  # human boxes win on anchor frames
            res[f] = {k: list(v[:4]) + [1.0, 0] for k, v in anchors[f].items()}
        n = b - a + 1
        print(f"  frames {a}-{b}: {len(fs)} anchors, L {sum('L' in r for r in res.values())}/{n}, "
              f"R {sum('R' in r for r in res.values())}/{n} tracked, {len(flags)} flags, "
              f"{time.time() - t0:.0f}s", flush=True)
        return res, flags


def load_tracks(project, stem):
    p = Path(project) / "tracks" / f"{stem}.json"
    return json.loads(p.read_text()) if p.exists() else None


def save_tracks(project, stem, data):
    p = Path(project) / "tracks" / f"{stem}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, p)


def check(res, check_items):
    """IoU between the track and human boxes that were not used as anchors."""
    out = []
    for it in check_items:
        rec = res.get(it["frame"], {})
        for b in it["boxes"]:
            if b["cls"] == PENDING:
                continue
            best = max([box_iou(b["xyxy"], rec[s]) for s in rec] or [0.0])
            out.append({"id": it["id"], "frame": it["frame"], "iou": round(best, 3)})
    return out


def run_video(tr, store, readers, video, ranges, all_anchors):
    r = readers(video)
    stem = Path(video).stem
    anchors, chk = pick_anchors(store.items, video, ranges, all_anchors)
    if not anchors:
        print(f"{video}: no done item with boxes in {ranges} -> add an anchor first "
              f"(python -m gripper.prepare anchor --video {video} --frames 0, then draw the boxes)")
        return
    boxes = {it["frame"]: [b["xyxy"] for b in it["boxes"] if b["cls"] != PENDING] for it in anchors}
    data = load_tracks(store.dir, stem) or {"video": video, "fps": r.fps, "w": r.w, "h": r.h,
                                             "ranges": [], "anchors": {}, "frames": {}, "flags": [], "check": []}
    print(f"{video}: {len(ranges)} range(s), {len(anchors)} anchors, {len(chk)} held-out boxes for checking")
    for a, b in ranges:
        res, flags = tr.track_range(r, a, b, boxes)
        if not res:
            print(f"  frames {a}-{b}: no anchor inside, skipped")
            continue
        data["frames"] = {k: v for k, v in data["frames"].items() if not a <= int(k) <= b}
        data["frames"].update({str(f): v for f, v in res.items()})
        data["flags"] = [x for x in data["flags"] if not a <= x[0] <= b] + flags
        data["ranges"] = sorted([x for x in data["ranges"] if x != [a, b]] + [[a, b]])
        for f in boxes:
            if a <= f <= b:
                data["anchors"][str(f)] = res[f]
        save_tracks(store.dir, stem, data)
    c = check({int(k): v for k, v in data["frames"].items()}, chk)
    data["check"] = c
    save_tracks(store.dir, stem, data)
    if c:
        ious = np.array([x["iou"] for x in c])
        bad = [x for x in c if x["iou"] < 0.5]
        print(f"  check vs {len(c)} human boxes: median IoU {np.median(ious):.2f}, "
              f"{(ious >= 0.5).mean():.0%} >= 0.5" + (f"; low: {[x['id'] for x in bad][:8]}" if bad else ""))
        report(store, [stem])


def report(store, stems=None, max_imgs=24):
    """Why is the IoU low? For every held-out human box: the matched track box, IoU, area ratio and
    centre offset (in units of the human box size). 'extent' = same place, different size (box
    convention, harmless for the crops); 'off' = the track is somewhere else (a real tracking error).
    Writes <project>/track_check/<stem>/*.jpg and a contact sheet <stem>_sheet.jpg of the worst ones."""
    out_dir = store.dir / "track_check"
    d = store.dir / "tracks"
    for p in sorted(d.glob("*.json")):
        if stems and p.stem not in stems:
            continue
        t = json.loads(p.read_text())
        checked = {c["id"] for c in t.get("check", [])}
        rows = []
        for it in store.items:
            if it["id"] not in checked:
                continue
            rec = t["frames"].get(str(it["frame"]), {})
            for b in it["boxes"]:
                if b["cls"] == PENDING:
                    continue
                h = b["xyxy"]
                hc = np.array([(h[0] + h[2]) / 2, (h[1] + h[3]) / 2])
                size = max(1.0, area(h)) ** 0.5
                best = None
                for side, v in rec.items():
                    c = np.array([(v[0] + v[2]) / 2, (v[1] + v[3]) / 2])
                    m = (box_iou(h, v), -np.linalg.norm(c - hc) / size, side, v)
                    best = max(best, m) if best else m
                if best is None:
                    rows.append((it, b, None, 0.0, None, None, "off"))
                    continue
                iou, negoff, side, v = best
                ratio = area(v) / max(1.0, area(h))
                kind = "ok" if iou >= 0.5 else ("extent" if -negoff < 0.35 else "off")
                rows.append((it, b, side, iou, ratio, -negoff, kind))
        if not rows:
            print(f"{p.stem}: no held-out human boxes")
            continue
        from collections import Counter
        cnt = Counter(r[6] for r in rows)
        ratios = [r[4] for r in rows if r[4] is not None]
        offs = [r[5] for r in rows if r[5] is not None]
        print(f"{p.stem}: {len(rows)} boxes  ok={cnt['ok']} extent={cnt['extent']} off={cnt['off']}  "
              f"median track/human area {np.median(ratios):.2f}, median centre offset {np.median(offs):.2f}")
        bad = sorted([r for r in rows if r[6] != "ok"], key=lambda r: r[3])
        od = out_dir / p.stem
        od.mkdir(parents=True, exist_ok=True)
        tiles = []
        for it, b, side, iou, ratio, off, kind in bad[:max_imgs]:
            img = cv2.imread(str(store.frame_path(it)))
            if img is None:
                continue
            for s2, v in t["frames"].get(str(it["frame"]), {}).items():
                col = (255, 160, 0) if s2 == "L" else (0, 140, 255)
                cv2.rectangle(img, tuple(map(int, v[:2])), tuple(map(int, v[2:4])), col, 3)
                cv2.putText(img, f"track {s2}", (int(v[0]) + 4, int(v[3]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2)
            h = b["xyxy"]
            cv2.rectangle(img, tuple(map(int, h[:2])), tuple(map(int, h[2:4])), (0, 230, 0), 3)
            txt = (f"{it['id']}  {kind}  IoU {iou:.2f}" +
                   (f"  area x{ratio:.2f}  offset {off:.2f}" if ratio is not None else "  no track box"))
            cv2.rectangle(img, (0, 0), (img.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(img, txt, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.imwrite(str(od / f"{it['frame']:06d}_{kind}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            tiles.append(cv2.resize(img, (640, 360)))
        if tiles:
            while len(tiles) % 3:
                tiles.append(np.zeros_like(tiles[0]))
            sheet = np.vstack([np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)])
            cv2.imwrite(str(out_dir / f"{p.stem}_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
            print(f"  green = human box, blue/orange = track L/R -> {out_dir}/{p.stem}_sheet.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["segments", "video", "report"])
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--videos", nargs="*", default=None)
    ap.add_argument("--sam2", default="base_plus", choices=list(SAM2_MODELS))
    ap.add_argument("--ckpt", default=None, help="override the SAM 2 checkpoint path")
    ap.add_argument("--device", default=None)
    ap.add_argument("--window", type=int, default=50, help="frames per SAM 2 window (re-prompted with a box)")
    ap.add_argument("--all-anchors", action="store_true", help="use every done item as an anchor")
    a = ap.parse_args()

    store, readers = Store(a.project), Readers(a.data)
    if a.cmd == "report":
        report(store, [Path(v).stem for v in a.videos] if a.videos else None)
        return
    tr = SamTracker(a.sam2, a.device, a.ckpt, a.window, tmp=store.dir)
    if a.cmd == "segments":
        for v in a.videos or sorted(store.segments):
            if is_eval(v):
                continue
            run_video(tr, store, readers, v, [[s, e - 1] for s, e in store.segments[v]], a.all_anchors)
    else:
        if not a.videos:
            raise SystemExit("video: pass --videos")
        for v in a.videos:
            r = readers(v)
            run_video(tr, store, readers, v, [[0, r.n_meta - 1]], True)
    shutil.rmtree(tr.tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
