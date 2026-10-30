"""Browser-based box labeling tool for gripper_open / gripper_closed / gripper_unknown.

Runs on the machine that has the videos (the cluster); open it from a laptop through an SSH tunnel:

    python -m gripper.label_tool --port 8765            # on the cluster, from video-annotation/
    ssh -N -L 8765:localhost:8765 <cluster>             # on the laptop (add -J for a compute node)
    open http://localhost:8765

Labels: gripper_open / gripper_closed / gripper_unknown, plus "gripper" = gripper located but state
not decided yet (YOLOE pre-labels arrive like this). Fix the box, then give it a state; a frame with
a "gripper" box left cannot be marked done.

Images are the lossless full-resolution PNGs written by gripper.prepare (1280x720, identical to the
decoded video frame). Boxes are stored in original pixel coordinates in <project>/items.json.
Standard library only (+ OpenCV for the context "peek" frames and for adding neighbour frames).

Second page, http://localhost:8765/state : open/closed TIMELINE labelling of SAM 2 gripper tracks
(gripper.track). One unit = one gripper (L or R) over one tracked range; you scrub through the
gripper crops and mark the frames where the state changes. See gripper/README.md.
"""
import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2

from .common import CLASSES, DATA, LABELS, PENDING, PROJECT, FrameReader, Readers, Store, crop
from .timeline import States, changes_from_pred, load_all_tracks, parse_key, units

HTML = Path(__file__).with_name("label_tool.html")
HTML_STATE = Path(__file__).with_name("label_state.html")


class Cached:
    """A JSON file (or folder of JSON files) re-read when it changes on disk."""

    def __init__(self, path, loader):
        self.path, self.loader, self.stamp, self.val = Path(path), loader, None, None

    def get(self):
        p = self.path
        st = (max([x.stat().st_mtime for x in p.glob("*.json")] or [0]) if p.is_dir()
              else p.stat().st_mtime if p.exists() else 0)
        if st != self.stamp:
            self.val, self.stamp = self.loader(), st
        return self.val


class App:
    def __init__(self, project, data):
        self.store = Store(project)
        self.readers = Readers(data)
        self.lock = threading.Lock()
        self.data = data
        missing = self.store.ensure_frames(self.readers)
        if missing:
            self.store.save()
            print(f"extracted {missing} missing frames")
        self.mtime = self._items_mtime()
        self.tracks = Cached(Path(project) / "tracks", lambda: load_all_tracks(project))
        self.preds = Cached(Path(project) / "pred", lambda: _load_preds(project))
        self.states = States(project)
        self.crop_dir = Path(project) / "crops"
        self.pregen = set()

    def _items_mtime(self):
        return self.store.path.stat().st_mtime if self.store.path.exists() else 0

    def maybe_reload(self):
        """items.json edited by a command line tool (e.g. prepare anchor) while the tool runs."""
        with self.lock:
            if self._items_mtime() != self.mtime:
                self.store = Store(self.store.dir)
                if self.store.ensure_frames(self.readers):
                    self.store.save()
                self.mtime = self._items_mtime()

    def summary(self):
        self.maybe_reload()
        return [{k: it[k] for k in ("id", "video", "frame", "t", "status", "source")} |
                {"n": len(it["boxes"]), "human": all(b["by"] == "human" for b in it["boxes"]),
                 "pending": sum(b["cls"] == PENDING for b in it["boxes"])}
                for it in self.store.items]

    def get(self, iid):
        return self.store.items[self.store.index[iid]]

    def put(self, iid, body):
        with self.lock:
            it = self.get(iid)
            boxes = []
            for b in body["boxes"]:
                x1, y1, x2, y2 = (float(v) for v in b["xyxy"])
                if b["cls"] not in LABELS or x2 - x1 < 2 or y2 - y1 < 2:
                    continue
                boxes.append({"cls": b["cls"], "xyxy": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
                              "by": b.get("by", "human"),
                              **({"conf": b["conf"]} if "conf" in b and b.get("by") != "human" else {})})
            if body.get("status") == "done" and any(b["cls"] == PENDING for b in boxes):
                raise ValueError(f"'{PENDING}' boxes still need a state (open/closed/unknown) or deletion")
            it["boxes"] = boxes
            if body.get("status") in ("todo", "done", "skip"):
                it["status"] = body["status"]
                if it["status"] == "done":  # a reviewed frame: every remaining box is a human label
                    for b in boxes:
                        b["by"] = "human"
                        b.pop("conf", None)
            if "note" in body:
                it["note"] = body["note"]
            self.store.save()
            self.mtime = self._items_mtime()
            return it

    def add(self, body):
        with self.lock:
            src = self.get(body["from"])
            r = self.readers(src["video"])
            f = int(body["frame"])
            if not 0 <= f < r.n_meta:
                raise ValueError("frame out of range")
            it = self.store.add(src["video"], f, r.fps, "manual", seg=src.get("seg"),
                                boxes=[dict(b, by="copied") for b in src["boxes"]])
            if it is None:  # already exists
                return self.get(f"{Path(src['video']).stem}/f{f:06d}")
            # keep neighbour frames next to their source in the list
            self.store.items.remove(it)
            pos = self.store.items.index(src) + (1 if f > src["frame"] else 0)
            self.store.items.insert(pos, it)
            self.store.index = {x["id"]: k for k, x in enumerate(self.store.items)}
            self.store.ensure_frames(self.readers, [it])
            self.store.save()
            self.mtime = self._items_mtime()
            return it

    # ---------------------------------------------------------------- timeline page
    def unit_list(self):
        tr, pr = self.tracks.get(), self.preds.get()
        out = []
        for key, video, a, b, side in units(tr):
            stem = Path(video).stem
            fr = tr[stem]["frames"]
            st = self.states.get(key)
            out.append({"key": key, "video": video, "side": side, "a": a, "b": b,
                        "tracked": sum(1 for f in range(a, b + 1) if side in fr.get(str(f), {})),
                        "flags": sum(1 for x in tr[stem]["flags"] if a <= x[0] <= b and x[1] in (side, "LR")),
                        "done": st.get("done", False), "changes": len(st.get("changes", [])),
                        "pred": stem in pr})
        return out

    def unit(self, key):
        stem, a, b, side = parse_key(key)
        t = self.tracks.get()[stem]
        pr = self.preds.get().get(stem)
        frames = {f: t["frames"][str(f)][side] for f in range(a, b + 1)
                  if side in t["frames"].get(str(f), {})}
        pred = None
        if pr:
            pred = {f: pr["frames"][str(f)][side] for f in range(a, b + 1)
                    if side in pr["frames"].get(str(f), {})}
        st = self.states.get(key)
        model_changes = []
        if not st.get("changes") and pred:
            model_changes = changes_from_pred({str(k): v for k, v in pred.items()}, a, b)
        self._start_pregen(stem, t, a, b)
        return {"key": key, "video": t["video"], "stem": stem, "side": side, "a": a, "b": b, "fps": t["fps"],
                "frames": frames, "pred": pred, "pred_model": pr["model"] if pr else None,
                "flags": [x for x in t["flags"] if a <= x[0] <= b and x[1] in (side, "LR")],
                "anchors": sorted(int(f) for f in t["anchors"] if a <= int(f) <= b),
                "changes": st.get("changes", []) or model_changes, "bad": st.get("bad", []),
                "done": st.get("done", False), "classes": CLASSES}

    def put_unit(self, body):
        return self.states.put(body["key"], {"changes": body.get("changes", []), "bad": body.get("bad", []),
                                             "done": bool(body.get("done"))})

    def _crop_path(self, stem, side, f, box):
        h = "_".join(str(int(round(v))) for v in box[:4])
        return self.crop_dir / stem / side / f"f{f:06d}_{h}.jpg"

    def crop_bytes(self, stem, side, f):
        t = self.tracks.get()[stem]
        rec = t["frames"].get(str(f), {}).get(side)
        if not rec:
            return None
        p = self._crop_path(stem, side, f, rec)
        if not p.exists():
            img = self.readers(t["video"]).get(f)
            if img is None:
                return None
            p.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(p), crop(img, rec[:4]), [cv2.IMWRITE_JPEG_QUALITY, 92])
        return p.read_bytes()

    def _start_pregen(self, stem, t, a, b):
        """Cut all crops of a range in the background (sequential decode, its own reader), so that
        scrubbing / playing on the timeline page does not wait for video seeks."""
        k = (stem, a, b)
        if k in self.pregen:
            return
        self.pregen.add(k)

        def work():
            try:
                r = FrameReader(self.readers(t["video"]).path)
                for f in range(a, b + 1):
                    rec = t["frames"].get(str(f), {})
                    todo = [(s, v) for s, v in rec.items() if not self._crop_path(stem, s, f, v).exists()]
                    if not todo:
                        continue
                    img = r.get(f)
                    if img is None:
                        break
                    for s, v in todo:
                        p = self._crop_path(stem, s, f, v)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(p), crop(img, v[:4]), [cv2.IMWRITE_JPEG_QUALITY, 92])
            finally:
                self.pregen.discard(k)
        threading.Thread(target=work, daemon=True).start()

    def context(self, stem, f, side):
        t = self.tracks.get()[stem]
        img = self.readers(t["video"]).get(f)
        if img is None:
            return None
        for s, v in t["frames"].get(str(f), {}).items():
            col = (255, 160, 0) if s == "L" else (0, 140, 255)
            x1, y1, x2, y2 = map(int, v[:4])
            cv2.rectangle(img, (x1, y1), (x2, y2), col, 4 if s == side else 2)
            cv2.putText(img, s, (x1 + 4, max(24, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes()

    def add_anchor(self, body):
        """Add frame f of a tracked video as an anchor item; tracked boxes are copied in as 'gripper'."""
        stem, f = body["stem"], int(body["frame"])
        t = self.tracks.get()[stem]
        with self.lock:
            iid = f"{stem}/f{f:06d}"
            if iid in self.store.index:
                return self.get(iid)
            r = self.readers(t["video"])
            boxes = [{"cls": PENDING, "xyxy": v[:4], "by": "track"} for v in t["frames"].get(str(f), {}).values()]
            it = self.store.add(t["video"], f, r.fps, "anchor", boxes=boxes)
            self.store.ensure_frames(self.readers, [it])
            self.store.save()
            self.mtime = self._items_mtime()
            return it

    def peek(self, iid, offset):
        it = self.get(iid)
        img = self.readers(it["video"]).get(max(0, it["frame"] + offset))
        if img is None:
            return None
        return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()


def make_handler(app):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json", cache=False):
            if not isinstance(body, bytes):
                body = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "max-age=86400" if cache else "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            u = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(u.query).items()}
            try:
                if u.path == "/":
                    html = (HTML.read_text().replace("__LABELS__", json.dumps(LABELS))
                            .replace("__PENDING__", json.dumps(PENDING)))
                    self._send(200, html.encode(), "text/html; charset=utf-8")
                elif u.path == "/api/items":
                    self._send(200, app.summary())
                elif u.path == "/api/item":
                    self._send(200, app.get(q["id"]))
                elif u.path == "/img":
                    p = app.store.frame_path(app.get(q["id"]))
                    self._send(200, p.read_bytes(), "image/png", cache=True)
                elif u.path == "/state":
                    self._send(200, HTML_STATE.read_bytes(), "text/html; charset=utf-8")
                elif u.path == "/api/units":
                    self._send(200, app.unit_list())
                elif u.path == "/api/unit":
                    self._send(200, app.unit(q["key"]))
                elif u.path == "/crop":
                    b = app.crop_bytes(q["stem"], q["side"], int(q["f"]))
                    self._send(200, b, "image/jpeg", cache=True) if b else self._send(404, {"error": "no box"})
                elif u.path == "/ctx":
                    b = app.context(q["stem"], int(q["f"]), q.get("side"))
                    self._send(200, b, "image/jpeg") if b else self._send(404, {"error": "eof"})
                elif u.path == "/peek":
                    b = app.peek(q["id"], int(q["off"]))
                    self._send(200, b, "image/jpeg", cache=True) if b else self._send(404, {"error": "eof"})
                else:
                    self._send(404, {"error": "not found"})
            except (KeyError, ValueError) as e:
                self._send(400, {"error": repr(e)})

        def do_POST(self):
            u = urlparse(self.path)
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            try:
                if u.path == "/api/item":
                    self._send(200, app.put(body["id"], body))
                elif u.path == "/api/add":
                    self._send(200, app.add(body))
                elif u.path == "/api/unit":
                    self._send(200, app.put_unit(body))
                elif u.path == "/api/anchor":
                    self._send(200, app.add_anchor(body))
                else:
                    self._send(404, {"error": "not found"})
            except (KeyError, ValueError) as e:
                self._send(400, {"error": repr(e)})

    return H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--host", default="127.0.0.1", help="keep 127.0.0.1 and use an SSH tunnel")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    app = App(a.project, a.data)
    print(f"{len(app.store.items)} items in {a.project}/items.json")
    print(f"serving on http://{a.host}:{a.port}  (Ctrl+C to stop; every edit is saved immediately)")
    ThreadingHTTPServer((a.host, a.port), make_handler(app)).serve_forever()


def _load_preds(project):
    out = {}
    d = Path(project) / "pred"
    for p in sorted(d.glob("*.json")) if d.exists() else []:
        try:
            out[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError:
            pass
    return out


if __name__ == "__main__":
    main()
