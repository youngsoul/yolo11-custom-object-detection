#!/usr/bin/env python3
"""
Evaluate a trained YOLOv11 detection model on a folder of class-organized images.

Expected test data layout:
  <test_root>/
    <class_1>/images/*.jpg|*.jpeg|*.png
    <class_2>/images/*.jpg|*.jpeg|*.png
    ...

For each image, the expected class is the name of its parent class directory. The
script runs detection and assigns the image the predicted class equal to the most
confident detection's class (if any). It then compares the predicted class to the
expected class and reports overall/per-class accuracy and a confusion matrix.

Usage examples:
  python test_yolov11.py --model yolo11n_custom/yolo11n_custom_run1/weights/best.pt \
                         --test-root ./modeling_data/test \
                         --imgsz 640 --device mps --batch 16 --conf 0.25 --iou 0.7 \
                         --save-csv results.csv --save-confusion test_confusion.csv \
                         --save-misclassified ./test_misclassified \
                         --test-images-boxes ./test_images_with_boxes

Notes:
- If an image has no detections above the confidence threshold, the predicted class is "__none__".
- The model's internal class names (`model.names`) are used to map detection class indices to strings.
- Only the top-confidence detection per image is used to assign a single predicted class label.
"""

import argparse
import csv
import os
import shutil
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from ultralytics import YOLO
import cv2

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
NONE_CLASS = "__none__"  # used when no detection is found


@dataclass
class ImageRecord:
    path: Path
    expected: str


@dataclass
class Prediction:
    path: Path
    expected: str
    predicted: str
    confidence: float

    @property
    def correct(self) -> bool:
        return self.expected == self.predicted


def discover_classes_and_images(test_root: Path) -> Tuple[List[str], List[ImageRecord]]:
    classes: List[str] = []
    records: List[ImageRecord] = []

    if not test_root.exists():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    for entry in sorted(test_root.iterdir()):
        if not entry.is_dir() or entry.name.startswith('.'):
            continue
        cls = entry.name
        classes.append(cls)
        images_dir = entry / "images"
        if images_dir.exists():
            for p in sorted(images_dir.rglob('*')):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    records.append(ImageRecord(path=p, expected=cls))
    if not classes:
        raise RuntimeError(f"No class subdirectories found in {test_root}")
    if not records:
        raise RuntimeError(f"No images found under any '<class>/images' in {test_root}")
    return classes, records


def run_inference(model: YOLO, records: List[ImageRecord], imgsz: int, device: str,
                  conf: float, iou: float, batch: int) -> Tuple[List[Prediction], List]:
    # Ultralytics supports list of paths with batch inference
    img_paths = [str(r.path) for r in records]
    results = model.predict(source=img_paths, imgsz=imgsz, device=device, conf=conf, iou=iou,
                            batch=batch, verbose=False)

    # Map class indices to names
    id2name: Dict[int, str] = {}
    # model.names can be dict or list
    if isinstance(model.names, dict):
        id2name = {int(k): str(v) for k, v in model.names.items()}
    else:
        id2name = {i: str(name) for i, name in enumerate(model.names)}

    preds: List[Prediction] = []
    for rec, res in zip(records, results):
        # Each res contains .boxes with .cls and .conf tensors; select top confidence
        pred_class = NONE_CLASS
        pred_conf = 0.0
        try:
            boxes = res.boxes  # may be empty
            if boxes is not None and len(boxes) > 0:
                confs = boxes.conf.cpu().tolist()
                clss = boxes.cls.cpu().tolist()
                # find best detection by confidence
                best_idx = max(range(len(confs)), key=lambda i: confs[i])
                best_cls_id = int(clss[best_idx])
                pred_class = id2name.get(best_cls_id, str(best_cls_id))
                pred_conf = float(confs[best_idx])
        except Exception:
            # If parsing fails for any reason, keep NONE_CLASS
            pass
        preds.append(Prediction(path=rec.path, expected=rec.expected, predicted=pred_class, confidence=pred_conf))
    return preds, results


def compute_metrics(classes: List[str], preds: List[Prediction]):
    # Include NONE_CLASS as a possible predicted class in confusion
    all_pred_labels = classes + [NONE_CLASS]

    # Confusion matrix counts: expected -> predicted -> count
    confusion: Dict[str, Counter] = {c: Counter() for c in classes}

    per_class_totals: Counter = Counter()
    per_class_correct: Counter = Counter()
    none_count = 0

    for p in preds:
        per_class_totals[p.expected] += 1
        if p.correct:
            per_class_correct[p.expected] += 1
        if p.predicted == NONE_CLASS:
            none_count += 1
        # record confusion
        if p.predicted not in all_pred_labels:
            # if model predicted a name that isn't one of our expected classes,
            # still record it for completeness
            confusion.setdefault(p.expected, Counter())[p.predicted] += 1
            if p.predicted not in all_pred_labels:
                all_pred_labels.append(p.predicted)
        else:
            confusion[p.expected][p.predicted] += 1

    total = len(preds)
    correct_total = sum(1 for p in preds if p.correct)
    overall_acc = correct_total / total if total else 0.0

    per_class_acc: Dict[str, float] = {}
    for c in classes:
        n = per_class_totals[c]
        per_class_acc[c] = (per_class_correct[c] / n) if n else 0.0

    return {
        "overall_acc": overall_acc,
        "total": total,
        "correct_total": correct_total,
        "none_count": none_count,
        "per_class_totals": per_class_totals,
        "per_class_correct": per_class_correct,
        "per_class_acc": per_class_acc,
        "confusion": confusion,
        "labels": all_pred_labels,
    }


def print_report(classes: List[str], metrics: Dict, preds: List[Prediction]) -> None:
    print("YOLOv11 Test Report")
    print("-" * 60)
    print(f"Images evaluated: {metrics['total']}")
    print(f"Overall accuracy: {metrics['overall_acc']*100:.2f}%  (correct={metrics['correct_total']})")
    print(f"No-detection images: {metrics['none_count']}")
    print()
    print("Per-class accuracy:")
    for c in classes:
        total = metrics['per_class_totals'][c]
        acc = metrics['per_class_acc'][c]
        correct = metrics['per_class_correct'][c]
        print(f"  {c:20s} acc={acc*100:6.2f}%  correct={correct:4d}/{total:4d}")
    print()
    # Compact confusion matrix
    labels = metrics['labels']
    print("Confusion matrix (expected -> predicted counts):")
    header = ["expected\\pred"] + labels
    print("\t".join(header))
    for exp in classes:
        row = [exp]
        row += [str(metrics['confusion'][exp].get(pred, 0)) for pred in labels]
        print("\t".join(row))
    print()


def save_csv(preds: List[Prediction], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "expected", "predicted", "confidence", "correct"])
        for p in preds:
            w.writerow([str(p.path), p.expected, p.predicted, f"{p.confidence:.6f}", int(p.correct)])


def save_confusion_csv(classes: List[str], metrics: Dict, path: Path) -> None:
    labels = metrics['labels']
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["expected\\predicted"] + labels)
        for exp in classes:
            row = [exp] + [metrics['confusion'][exp].get(pred, 0) for pred in labels]
            w.writerow(row)


def copy_misclassified(preds: List[Prediction], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in preds:
        if not p.correct:
            # subdir: expected__predicted
            sub = out_dir / f"{p.expected}__{p.predicted}"
            sub.mkdir(parents=True, exist_ok=True)
            dst = sub / p.path.name
            try:
                shutil.copy2(p.path, dst)
            except Exception:
                pass


def save_annotated_images(records: List[ImageRecord], results: List, out_root: Path) -> None:
    """Save images with predicted bounding boxes drawn.

    The output directory will mirror the input structure: <out_root>/<expected>/images/<filename>.
    """
    out_root.mkdir(parents=True, exist_ok=True)
    for rec, res in zip(records, results):
        try:
            annotated = res.plot()  # BGR numpy array with boxes/labels drawn
            out_dir = out_root / rec.expected / "images"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / rec.path.name
            cv2.imwrite(str(out_path), annotated)
        except Exception:
            # Skip silently on write/plot errors
            continue


def parse_args(argv: List[str] | None = None):
    ap = argparse.ArgumentParser(description="Evaluate a YOLOv11 model on class-organized test images.")
    ap.add_argument("--model", required=True, type=str, help="Path to trained YOLO model weights (e.g., best.pt)")
    ap.add_argument("--test-root", required=True, type=Path, help="Path to test root folder with <class>/images structure")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size (default: 640)")
    ap.add_argument("--device", type=str, default="cpu", help="Device for inference: cpu, mps, 0, 0,1, etc.")
    ap.add_argument("--batch", type=int, default=16, help="Batch size for inference (default: 16)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    ap.add_argument("--iou", type=float, default=0.7, help="IoU threshold for NMS (default: 0.7)")
    ap.add_argument("--save-csv", type=Path, default=None, help="Optional path to save per-image predictions CSV")
    ap.add_argument("--save-confusion", type=Path, default=None, help="Optional path to save confusion matrix CSV")
    ap.add_argument("--save-misclassified", type=Path, default=None, help="Optional folder to copy misclassified images")
    ap.add_argument("--test-images-boxes", type=Path, default=None, help="If set, save annotated images with bounding boxes under this directory, preserving <class>/images structure")
    return ap.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    model = YOLO(args.model)

    classes, records = discover_classes_and_images(args.test_root)

    preds, results = run_inference(model=model,
                          records=records,
                          imgsz=args.imgsz,
                          device=args.device,
                          conf=args.conf,
                          iou=args.iou,
                          batch=args.batch)

    metrics = compute_metrics(classes, preds)

    print_report(classes, metrics, preds)

    if args.save_csv:
        save_csv(preds, Path(args.save_csv))
        print(f"Saved per-image CSV to: {args.save_csv}")
    if args.save_confusion:
        save_confusion_csv(classes, metrics, Path(args.save_confusion))
        print(f"Saved confusion matrix CSV to: {args.save_confusion}")
    if args.save_misclassified:
        copy_misclassified(preds, Path(args.save_misclassified))
        print(f"Copied misclassified images under: {args.save_misclassified}")
    if args.test_images_boxes:
        save_annotated_images(records, results, Path(args.test_images_boxes))
        print(f"Saved annotated images with boxes under: {args.test_images_boxes}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
