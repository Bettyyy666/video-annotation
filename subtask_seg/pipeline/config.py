"""All tunable constants in one place."""

FPS_OUT = 50.0               # frame numbers in the output JSON: f = round(t * 50)

# ---- perception ----
# gripper boxes / open-closed come from ../gripper (SAM 2 tracks + crop classifier).
# YOLO-World is optional (annotate.py --objects); the alignment no longer needs object boxes.
DET_MODEL = "yolov8x-worldv2.pt"
DET_CONF = 0.08
OBJ_STRIDE = 2
DETAIL_VOCAB = {
    "garment": ["t-shirt", "pants", "shorts", "shirt", "sweater"],
}

# ---- [A] gripper state -> grasp episodes ----   (signal: an = 1 - p_closed, NaN = unknown)
CLS_MIN_CONF = 0.8           # max(p_open, p_closed) (renormalised without 'unknown') below this -> unknown
PARK_S = 1.0                 # a closure is a grasp only if the gripper moves within PARK_S before / after it ...
PARK_K = 0.5                 # ... by more than PARK_K x its box diagonal (0 = off). Parked grippers are skipped
MAX_INTERP_S = 0.2           # fill 'unknown' runs up to this long; longer gaps = state unknown (held)
CLOSE_T = 0.35               # closed below this ...
OPEN_T = 0.55                # ... open again above this (hysteresis)
MIN_CLOSED_S = 0.16          # ignore closures shorter than this
MERGE_GAP_S = 0.10           # merge closures separated by flicker shorter than this
ONSET_DELTA = 0.04           # "starts opening" = signal rises this much above the closed plateau
ONSET_SHIFT = 0              # frames subtracted from every detected release: classifier lag, measured by
                             # tools/measure_lag.py on the gripper timeline labels (annotate.py --onset-shift)
D_OBJ = 25.0                 # px: object moved -> it was picked up (only with --objects)
D_GRIP = 60.0                # px: gripper travelled while closed -> carried something
FAIL_MAX_S = 4.0             # a missed grasp is short ...
RETRY_GAP_S = 3.0            # ... and the same gripper closes again at the same spot within this time
ASSOC_K = 0.75               # "same spot" = within ASSOC_K * gripper-box diagonal
DEPART_HOLD = 5

# ---- [B] VLM scorer (Qwen3-VL-8B, multiple choice / yes-no log-probs) ----
VLM_MODEL = "/users/rzhou52/data/users/rzhou52/Qwen3-VL-8B-Instruct"
VLM_WIDTH = 960              # px, frames given to the VLM (from the 1280x720 originals)
CLIP_S = 2.0                 # one clip = 2 s ...
CLIP_STEP_S = 1.0            # ... every 1 s
CLIP_FRAMES = 6

# ---- [C] grammar DP ----
VLM_TEMP = 1.0               # log-probs / VLM_TEMP, renormalised (VLM is over-confident per clip)
VLM_FLOOR = -99.0            # per-clip log-prob floor
VLM_CENTER = True            # subtract each option's mean over the video
VLM_W = 1.0                  # weight of the (option-mean-centred) VLM log-probs
MIN_SEG_S = 0.5              # shortest subtask
MOTION_SPEED = 3.0           # px/frame, gripper box centre: above = moving ...
MOTION_STILL_S = 0.3         # ... after at least this long below MOTION_SPEED / 2 -> "starts moving" candidate
C_CAND_RELEASE = 0.0         # end-boundary cost by candidate type, for rows that end with a release
C_CAND_MOTION_FOR_REL = 2.0
C_CAND_GRID = 3.0            # 1-s grid fallback (no gripper event found there)
C_CAND_RELEASE_FOR_MOVE = 0.5  # rows that do not end with a release: a release is an OK end point too
C_CAND_MOTION = 0.0
C_SKIP_MAND = 6.0            # skipping a row occurrence the task sheet requires (scaled by the VLM count)
C_SKIP_OPT = 0.0             # skipping an occurrence beyond the task sheet minimum (no VLM count) ...
C_EXTRA_OCC = 0.5            # ... and using it
C_COND = 4.0                 # conditional rows: use costs C_COND * (1 - p_yes), skip costs C_COND * p_yes
PER_ITEM_DEFAULT = (1, 8)    # count range for "one per item" rows when no count is available
PER_ITEM_SLACK = 2

# ---- [D] retry ----
LIFT_P = 0.5                 # VLM p(object lifted) below this confirms a missed grasp
MIN_MISS_S = 0.4             # closures shorter than this are classifier flicker, not grasp attempts
MIN_CARRY_S = 0.5            # re-grasp rule: the first grasp must be a real hold of at least this long ...
MIN_REGRASP_GAP_S = 0.3      # ... and the new grasp must start at least this long after the release
REACH_MAX_S = 3.0            # recovery_t: look back at most this long for the start of the final reach

# ---- self-check ----
TOL_FRAMES = 25              # ±0.5 s at 50 fps
