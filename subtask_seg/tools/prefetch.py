"""Download the YOLO-World weights and its CLIP text encoder once (login node), so GPU jobs need no network.
    cd video-annotation/subtask_seg && python tools/prefetch.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ultralytics import YOLOWorld  # noqa: E402

from pipeline import config as C  # noqa: E402

w = YOLOWorld(C.DET_MODEL)         # -> ./yolov8x-worldv2.pt
w.set_classes(["towel"])           # downloads the CLIP text encoder to ~/.cache/clip
print("prefetch ok")
