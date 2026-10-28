"""Small experiment: classify each gripper (left/right) per frame as
gripper_open / gripper_closed / gripper_unknown with YOLOE-11 (open-vocab YOLO11).

COCO-pretrained YOLO11 has no gripper class, so we use text prompts instead.
A side is "unknown" when no gripper is found, the score is low, or the
open-vs-closed scores for the same box are too close to call.

Usage:
    python yolo_exp1.py --device 0
"""
import argparse
import glob
import json
import os

import cv2
from ultralytics import YOLOE

YOLO_DIR = "/users/rzhou52/data/users/rzhou52/YOLO11/yoloe"
FRAMES = "/users/rzhou52/data/users/rzhou52/video-annotation/runs/exp1/frames"
PROMPTS = {"open robot gripper": "gripper_open", "closed robot gripper": "gripper_closed"}
COLORS = {"gripper_open": (0, 200, 0), "gripper_closed": (0, 0, 220), "gripper_unknown": (0, 200, 220)}


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def classify_side(dets, conf_thr, margin):
    """dets: list of (label, score, box) on one side of the image."""
    if not dets:
        return {"state": "gripper_unknown", "reason": "not_detected", "box": None, "scores": {}}
    best = max(dets, key=lambda d: d[1])
    scores = {lab: 0.0 for lab in PROMPTS.values()}
    for lab, s, box in dets:
        if iou(box, best[2]) > 0.5:
            scores[lab] = max(scores[lab], s)
    other = max(s for lab, s in scores.items() if lab != best[0])
    if best[1] < conf_thr:
        state, reason = "gripper_unknown", "low_conf"
    elif best[1] - other < margin:
        state, reason = "gripper_unknown", "ambiguous"
    else:
        state, reason = best[0], "ok"
    return {"state": state, "reason": reason, "box": [round(v, 1) for v in best[2]],
            "scores": {k: round(v, 3) for k, v in scores.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--weights", default=os.path.join(YOLO_DIR, "yoloe-11l-seg.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(FRAMES), "yolo_gripper"))
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--det-conf", type=float, default=0.05, help="raw detection threshold")
    ap.add_argument("--conf", type=float, default=0.15, help="min score to trust a state")
    ap.add_argument("--margin", type=float, default=0.05, help="min open-vs-closed score gap")
    args = ap.parse_args()

    # YOLOE looks for mobileclip_blt.ts in cwd
    os.chdir(YOLO_DIR)
    model = YOLOE(args.weights)
    model.set_classes(list(PROMPTS))

    os.makedirs(os.path.join(args.out, "vis"), exist_ok=True)
    results = []
    for path in sorted(glob.glob(os.path.join(args.frames, "*.jpg"))):
        r = model.predict(path, device=args.device, conf=args.det_conf, verbose=False)[0]
        W = r.orig_shape[1]
        sides = {"left": [], "right": []}
        for c, s, b in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxy.tolist()):
            side = "left" if (b[0] + b[2]) / 2 < W / 2 else "right"
            sides[side].append((PROMPTS[model.names[int(c)]], s, b))

        rec = {"frame": os.path.basename(path)}
        img = r.orig_img.copy()
        for side, dets in sides.items():
            rec[side] = classify_side(dets, args.conf, args.margin)
            if rec[side]["box"]:
                x1, y1, x2, y2 = map(int, rec[side]["box"])
                color = COLORS[rec[side]["state"]]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(img, f"{side}:{rec[side]['state'][8:]}", (x1 + 2, max(y1, 0) + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.imwrite(os.path.join(args.out, "vis", rec["frame"]), img)
        results.append(rec)
        print(f"{rec['frame']}  left={rec['left']['state']:<16} ({rec['left']['reason']})"
              f"  right={rec['right']['state']:<16} ({rec['right']['reason']})")

    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump({"args": vars(args), "frames": results}, f, indent=1)
    print("saved to", args.out)


if __name__ == "__main__":
    main()
