"""Load one task from task_sheets.json and expand it into an ordered list of subtask slots.

task_sheets.json format (version 2026-10-20):
  tasks[i] = {task, title, steps:[{n,title}], subtasks:[{step, subtask, object, count, ends_when}], notes:[...]}
  video "12_fold_four_towels" <-> tasks[11].task == "fold_four_towels"
  count: "1" | "4, one per box" | "0–1, only if ..." | "per glass" | "0–1 per plate, only if ..."
  repeated step blocks are declared in notes: "Steps 1 to 5 run once per cable, four times in all ..."
  last row: object "–", ends_when "Last frame"
"""
import json
import re
from dataclasses import dataclass, field
from typing import Optional

from . import config as C

NUM = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8}


@dataclass
class Row:
    step_n: int
    step_name: str
    name: str
    obj: Optional[str]
    count: Optional[tuple]  # (lo, hi) total over the episode; None = one per item in the scene
    per_item_cond: bool     # "0–1 per plate, only if ..." -> 0..n_items
    ends_when: str
    release: bool           # ends when a gripper releases / lets go
    regrasp_ok: bool        # task notes say re-grips here are normal, not retries
    count_text: str = ""    # raw count column ("0–1, only if a pile is already at the back-right")


@dataclass
class Step:
    step_n: int
    step_name: str
    rows: list = field(default_factory=list)


@dataclass
class Slot:
    row: Row
    round: int
    mandatory: bool


def find_task(path, episode_id):
    data = json.load(open(path))
    tasks = data["tasks"] if isinstance(data, dict) else data
    key = re.sub(r"^\d+_", "", episode_id)
    for t in tasks:
        if t.get("task") == key:
            return t
    m = re.match(r"(\d+)_", episode_id)               # fall back to the position in the list
    if m and 1 <= int(m.group(1)) <= len(tasks):
        return tasks[int(m.group(1)) - 1]
    raise KeyError(f"task '{episode_id}' not found in {path}")


def parse_count(spec):
    """-> (range or None, per_item_conditional)."""
    s = str(spec).strip().replace("–", "-").replace("—", "-")
    m = re.match(r"(\d+)\s*-\s*(\d+)\s*per\b", s)     # "0-1 per plate, only if ..."
    if m:
        return None, True
    m = re.match(r"(\d+)\s*-\s*(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(2))), False
    m = re.match(r"(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(1))), False
    return None, False                                # "per glass", "per item, front to back" ...


def _is_release(ends_when):
    if re.search(r"releas|let(s)? go", ends_when, re.I):
        return True
    return False


def _repeat_blocks(notes):
    """'Steps 1 to 5 run once per cable, four times ...' -> [(1, 5, 4 or None)]"""
    out = []
    for n in notes:
        m = re.search(r"Steps (\d+) to (\d+) (?:run|repeat)s? once per (\w+)", n)
        if m:
            k = re.search(r"\b(\w+) times", n)
            rounds = NUM.get(k.group(1).lower()) if k else None
            out.append((int(m.group(1)), int(m.group(2)), rounds))
    return out


def _regrasp_ok(notes, row, step_name, objects):
    """Row is covered by a notes sentence saying re-grips / extra grasps there are not retries."""
    sents = []
    for n in notes:
        for s in re.split(r"(?<=\.)\s+", n):
            if sents and re.match(r"So are\b", s):
                sents[-1] += " " + s
            else:
                sents.append(s)
    stop = {w.lower() for o in objects for w in o.split()} | {"the", "and", "with", "into", "onto", "from", "one",
                                                               "this", "that", "each", "half", "both"}
    words = {w for w in re.findall(r"[a-z]{4,}", (row + " " + step_name).lower()) if w not in stop}
    if any(re.search(r"\bis a retry", s, re.I) for s in sents):   # e.g. cutlery: re-grasping IS a retry
        return False
    for k, s in enumerate(sents):
        if re.search(r"not (a )?retr|are normal|is normal", s, re.I):
            ctx = (sents[k - 1] + " " + s if k else s).lower()      # "Straighten covers ... . Letting go ... normal"
            if any(re.search(r"\b" + w[:5], ctx) for w in words):
                return True
    return False


def load(path, episode_id):
    task = find_task(path, episode_id)
    notes = task.get("notes", []) or []
    steps = {s["n"]: Step(int(s["n"]), s["title"]) for s in task["steps"]}
    objects = sorted({r["object"] for r in task["subtasks"] if r["object"] not in ("–", "-", "")})
    for r in task["subtasks"]:
        st = steps[r["step"]]
        cnt, pic = parse_count(r["count"])
        obj = r["object"] if r["object"] not in ("–", "-", "") else None
        st.rows.append(Row(st.step_n, st.step_name, r["subtask"], obj, cnt, pic, r["ends_when"],
                           _is_release(r["ends_when"]),
                           _regrasp_ok(notes, r["subtask"], st.step_name, objects), str(r["count"])))
    steps = [steps[k] for k in sorted(steps)]
    return task, steps, objects, _repeat_blocks(notes)


def _per_item(obj, n_inst, conditional=False):
    n = n_inst.get(obj) if obj else None
    if not n:
        lo, hi = C.PER_ITEM_DEFAULT
    else:
        lo, hi = max(1, n - 1), n + C.PER_ITEM_SLACK
    return (0, hi) if conditional else (lo, hi)


def is_tail(row):
    return row.obj is None and re.search(r"last frame", row.ends_when, re.I) is not None


def expand(steps, repeats, n_inst):
    """Slots in task-sheet order (excluding the final 'Last frame' row). Slots beyond the minimum are optional.
    A repeated block (Steps a..b once per item) is laid out round by round."""
    slots = []
    by_n = {st.step_n: st for st in steps}
    done = set()
    for st in steps:
        if st.step_n in done:
            continue
        blk = next(((a, b, k) for a, b, k in repeats if a == st.step_n), None)
        if blk is None:
            for row in st.rows:
                if is_tail(row):
                    continue
                lo, hi = row.count if row.count else _per_item(row.obj, n_inst, row.per_item_cond)
                slots += [Slot(row, 0, k < lo) for k in range(hi)]
            done.add(st.step_n)
            continue
        a, b, k = blk
        block = [by_n[n] for n in range(a, b + 1) if n in by_n]
        first = block[0].rows[0]
        if k is None:                                  # rounds from the first row's count, else from the scene
            k = first.count[0] if first.count and first.count[0] == first.count[1] else None
        lo_r, hi_r = (k, k) if k else _per_item(first.obj, n_inst)
        for rd in range(hi_r):
            for bst in block:
                for row in bst.rows:
                    if row.count:                      # total count spread evenly over the rounds
                        lo, hi = (-(-row.count[0] // max(lo_r, 1)), -(-row.count[1] // max(lo_r, 1)))
                        lo, hi = min(lo, row.count[0]), min(hi, row.count[1])
                    else:
                        lo, hi = (0, 1) if row.per_item_cond else (1, 1)
                    slots += [Slot(row, rd, rd < lo_r and q < lo) for q in range(hi)]
        done.update(n for n in range(a, b + 1))
    # the final "Last frame" row (confirm / return home) absorbs whatever grasps happen after the last real
    # subtask (grippers often close again on the way home); skipping it is free, export always ends it at T-1
    tail = next((r for st in steps for r in st.rows if is_tail(r)), None)
    if tail is not None:
        slots.append(Slot(tail, 0, False))
    return slots
