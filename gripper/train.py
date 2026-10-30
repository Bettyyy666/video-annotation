"""Fine-tune YOLO11 (detect) on the 3 gripper classes at full resolution.

imgsz=1280 means the 1280x720 frame is fed at its native size (letterboxed to 1280x1280 for the
square mosaic batches) - no downscaling. Grippers are small and finger gaps are a few pixels wide,
so this matters more than the extra compute.

  python -m gripper.train --name r1                         # yolo11n
  python -m gripper.train --name r1s --weights $YOLO/yolo11s.pt
"""
import argparse

from .common import PROJECT, YOLO_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--data", default=None, help="default <project>/dataset/data.yaml")
    ap.add_argument("--weights", default=f"{YOLO_DIR}/yolo11n.pt")
    ap.add_argument("--name", default="r1")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    model.train(
        data=a.data or f"{a.project}/dataset/data.yaml",
        imgsz=a.imgsz, epochs=a.epochs, batch=a.batch, device=a.device, workers=a.workers,
        project=f"{a.project}/train", name=a.name, exist_ok=True,
        patience=40, cos_lr=True, close_mosaic=15,
        # augmentations: geometry kept mild (fixed top-down camera); left-right flip is fine because
        # the classes describe the state, not which arm it is
        degrees=0.0, shear=0.0, perspective=0.0, translate=0.1, scale=0.3, fliplr=0.5, flipud=0.0,
        hsv_h=0.015, hsv_s=0.5, hsv_v=0.35, mosaic=1.0, mixup=0.0,
        plots=True, seed=0, deterministic=False,
    )
    m = model.val(data=a.data or f"{a.project}/dataset/data.yaml", imgsz=a.imgsz, split="val",
                  device=a.device, plots=True, project=f"{a.project}/train", name=f"{a.name}_val",
                  exist_ok=True)
    print(f"mAP50={m.box.map50:.3f}  mAP50-95={m.box.map:.3f}")
    print(f"weights: {a.project}/train/{a.name}/weights/best.pt")


if __name__ == "__main__":
    main()
