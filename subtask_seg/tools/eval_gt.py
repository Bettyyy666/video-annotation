"""Compare a predicted episode JSON with a hand-made ground-truth JSON in the same (submission) format.

    python tools/eval_gt.py results/12_fold_four_towels.json ../data/12_fold_four_towels_GT.json
    python tools/eval_gt.py results ../data            # every <id>_GT.json in ../data with a result

Reports
  * subtask sequence: (name, occurrence) of prediction vs GT, aligned in order (difflib); missing / extra rows
  * boundaries: for every internal GT boundary (f_end of all but the last subtask) the signed error of the
    matched predicted subtask's f_end (pred - GT, frames) and of the nearest predicted boundary;
    median |e|, p90, share within ±25 frames (= ±0.5 s, the spec tolerance), signed mean (systematic lag)
  * steps: start / end error per step interval
  * status: ok / suboptimal / failed agreement on matched subtasks, recovery_t error
Writes <result>.eval.csv next to the prediction (one row per GT subtask).
"""
import argparse
import csv
import difflib
import glob
import json
import os

import numpy as np

TOL = 25


def load(p):
    return json.load(open(p))


def key(s):
    return f"{s['name']}#{s['occurrence']}"


def stats(err):
    e = np.array(err, float)
    if not len(e):
        return "n=0"
    ae = np.abs(e)
    return (f"n={len(e)}  median|e|={np.median(ae):.0f}f  p90={np.percentile(ae, 90):.0f}f  "
            f"max={ae.max():.0f}f  within±25f={np.mean(ae <= TOL):.0%}  signed mean={e.mean():+.1f}f")


def evaluate(pred_path, gt_path, quiet=False):
    P, G = load(pred_path), load(gt_path)
    ps, gs = P["subtasks"], G["subtasks"]
    out = {"video": G["episode"]["episode_id"]}
    say = (lambda *x: None) if quiet else print
    say(f"\n=== {out['video']}   outcome pred={P['episode']['outcome']} gt={G['episode']['outcome']}")
    sm = difflib.SequenceMatcher(a=[key(s) for s in gs], b=[key(s) for s in ps], autojunk=False)
    pairs, missing, extra = [], [], []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            pairs += list(zip(range(i1, i2), range(j1, j2)))
        else:
            missing += list(range(i1, i2))
            extra += list(range(j1, j2))
    say(f"subtasks: gt {len(gs)}, pred {len(ps)}, matched {len(pairs)}")
    for i in missing:
        say(f"  MISSING in pred: seq {gs[i]['seq']} {key(gs[i])}  ({gs[i]['t_start']}-{gs[i]['t_end']}s)")
    for j in extra:
        say(f"  EXTRA in pred:   seq {ps[j]['seq']} {key(ps[j])}  ({ps[j]['t_start']}-{ps[j]['t_end']}s)")

    pb = np.array([s["f_end"] for s in ps[:-1]])
    rows, e_match, e_near = [], [], []
    pmap = dict(pairs)
    for i, g in enumerate(gs):
        last = i == len(gs) - 1
        j = pmap.get(i)
        em = None if (j is None or last) else ps[j]["f_end"] - g["f_end"]
        en = None if (last or not len(pb)) else int(pb[np.argmin(np.abs(pb - g["f_end"]))] - g["f_end"])
        if em is not None:
            e_match.append(em)
        if en is not None:
            e_near.append(en)
        st_p = ps[j]["status"] if j is not None else None
        rec_e = None
        if j is not None and g.get("recovery_t") is not None and ps[j].get("recovery_t") is not None:
            rec_e = round(ps[j]["recovery_t"] - g["recovery_t"], 3)
        rows.append({"seq": g["seq"], "name": g["name"], "occurrence": g["occurrence"], "gt_f_end": g["f_end"],
                     "pred_f_end": ps[j]["f_end"] if j is not None else "", "err_matched": "" if em is None else em,
                     "err_nearest": "" if en is None else en, "gt_status": g["status"], "pred_status": st_p or "",
                     "gt_recovery_t": g.get("recovery_t"), "recovery_err_s": rec_e})
    say("boundaries (matched subtask f_end - GT):  " + stats(e_match))
    say("boundaries (nearest predicted boundary):  " + stats(e_near))
    for r in rows:
        if r["err_matched"] != "" and abs(r["err_matched"]) > TOL:
            say(f"  off by {r['err_matched']:+d}f ({r['err_matched'] / 50:+.2f}s): seq {r['seq']} "
                f"{r['name']} #{r['occurrence']} (GT end {r['gt_f_end']})")

    st = [(r["gt_status"], r["pred_status"]) for r in rows if r["pred_status"]]
    agree = sum(a == b for a, b in st)
    say(f"status agreement on matched subtasks: {agree}/{len(st)}" +
        "".join(f"\n  seq {r['seq']}: gt {r['gt_status']} pred {r['pred_status']}"
                for r in rows if r["pred_status"] and r["gt_status"] != r["pred_status"]))
    rec = [r["recovery_err_s"] for r in rows if r["recovery_err_s"] is not None]
    if rec:
        say(f"recovery_t error (s): {rec}")

    gst = {}
    for s in G["steps"]:
        gst.setdefault(s["step_n"], []).append(s)
    pst = {}
    for s in P["steps"]:
        pst.setdefault(s["step_n"], []).append(s)
    say("steps (start err, end err in frames):")
    for n in sorted(set(gst) | set(pst)):
        g, p = gst.get(n, []), pst.get(n, [])
        if len(g) != len(p):
            say(f"  step {n}: gt {len(g)} interval(s), pred {len(p)}")
            continue
        say("  step %d: " % n + ", ".join(f"{a['f_start'] - b['f_start']:+d}/"
                                         f"{a['f_end'] - b['f_end']:+d}" for a, b in zip(p, g)))
    with open(os.path.splitext(pred_path)[0] + ".eval.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out.update(e_match=e_match, e_near=e_near, missing=len(missing), extra=len(extra),
               status_agree=agree, status_n=len(st))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="prediction JSON or results dir")
    ap.add_argument("gt", help="GT JSON or dir with <id>_GT.json")
    a = ap.parse_args()
    if os.path.isdir(a.pred):
        res = []
        for g in sorted(glob.glob(os.path.join(a.gt, "*_GT.json"))):
            vid = os.path.basename(g)[:-len("_GT.json")]
            p = os.path.join(a.pred, vid + ".json")
            if os.path.exists(p):
                res.append(evaluate(p, g))
            else:
                print(f"\n=== {vid}: no prediction {p}")
        if res:
            print("\n=== ALL")
            print("boundaries (matched):  " + stats(sum((r["e_match"] for r in res), [])))
            print("boundaries (nearest):  " + stats(sum((r["e_near"] for r in res), [])))
            print(f"missing {sum(r['missing'] for r in res)}, extra {sum(r['extra'] for r in res)}, "
                  f"status {sum(r['status_agree'] for r in res)}/{sum(r['status_n'] for r in res)}")
    else:
        evaluate(a.pred, a.gt)


if __name__ == "__main__":
    main()
