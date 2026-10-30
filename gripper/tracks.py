"""Split per-frame gripper instances into a LEFT and a RIGHT track and derive p_open(t).

Simple nearest-neighbour association: each side keeps its last seen box centre; per frame we try
every assignment of the (at most 3) most confident instances to {L, R, none} and keep the cheapest.
A side that has not been seen for a while falls back to a weak prior (L in the left half, R in the
right half of the image). Good enough for top-down bimanual video where the arms rarely cross;
check the output plot if they do.
"""
import itertools
import math

MISSING = None


def _centre(b):
    return ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)


def assign_sides(per_frame, frame_ids, width, conf_min=0.25, max_jump=250, reset=50):
    """per_frame: list (one per processed frame) of instance lists from Detector.
    frame_ids: matching frame indices. Returns list of {"L": inst|None, "R": inst|None}."""
    prior_x = {"L": 0.25 * width, "R": 0.75 * width}
    last = {"L": None, "R": None}  # (frame, (cx, cy))
    out = []
    for f, insts in zip(frame_ids, per_frame):
        cands = [i for i in insts if i["conf"] >= conf_min][:3]
        opts = [None] + list(range(len(cands)))
        best, best_cost = {"L": None, "R": None}, math.inf
        for li, ri in itertools.product(opts, opts):
            if li is not None and li == ri:
                continue
            asg = {"L": None if li is None else cands[li], "R": None if ri is None else cands[ri]}
            cost, ok = 0.0, True
            for side, inst in asg.items():
                if inst is None:
                    continue
                c = _centre(inst["xyxy"])
                if last[side] and f - last[side][0] <= reset:
                    d = math.dist(c, last[side][1])
                    if d > max_jump * max(1.0, (f - last[side][0]) / 10):
                        ok = False  # implausible jump for this side
                else:
                    d = abs(c[0] - prior_x[side])
                cost += d - 1000  # reward assigning an instance
            if asg["L"] and asg["R"] and _centre(asg["L"]["xyxy"])[0] > _centre(asg["R"]["xyxy"])[0] + 150:
                cost += 400  # mild penalty for crossed arms
            if ok and cost < best_cost:
                best, best_cost = asg, cost
        for side in ("L", "R"):
            if best[side]:
                last[side] = (f, _centre(best[side]["xyxy"]))
        out.append(best)
    return out


def p_open(inst):
    """Probability-like open score of one instance, or None when the detector gives no state."""
    if inst is None:
        return MISSING
    s = inst["scores"]
    so, sc = s["gripper_open"], s["gripper_closed"]
    if so + sc <= 0 or s["gripper_unknown"] > max(so, sc):
        return MISSING
    return so / (so + sc)
