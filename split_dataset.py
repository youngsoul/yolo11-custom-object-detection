#!/usr/bin/env python3
"""
Split a YOLO-style dataset into train/val/test while keeping image/label pairs together.

Behavior per requirements:
- Takes a path to a source (root) directory containing many class subdirectories (e.g., bird, car, cat, ...).
- Takes a path to a target directory where the resulting files will be COPIED to.
- If the source directory does not exist, it will be created. Inside it, the subdirectories `train`, `val`, `test` will also be created.
  (Note: These are created to meet the specification and are ignored during class scanning.)
- For every class subdirectory found directly under the source directory (excluding `train`, `val`, `test`),
  the script creates the matching class directory structure in the target directory:
    target/
      ├── train/<class>/{images,labels}
      ├── val/<class>/{images,labels}
      └── test/<class>/{images,labels}
- From each source class directory, pairs of files with the same basename must be copied together:
    <basename>.jpg from images/ and <basename>.txt from labels/.
  Only basenames that have BOTH files present are considered. Pairs are split 80%/15%/5%
  to train/val/test respectively (rounding goes to test to ensure totals sum to N).

Usage:
  python split_dataset.py --source /path/to/source --target /path/to/target [--seed 42]

Assumptions:
- Images extension is strictly `.jpg` and labels extension is `.txt`, as requested.
- Source class structure:
    source/
      ├── class_a/
      │    ├── images/*.jpg
      │    └── labels/*.txt
      └── class_b/
           ├── images/*.jpg
           └── labels/*.txt

This script prints a concise summary of copies performed.
"""
# from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Set, Tuple

SPLITS = ("train", "val", "test")
IGNORED_SOURCE_DIRS = set(SPLITS)  # do not treat these as class names if they exist under source


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def find_classes(source: Path) -> List[str]:
    classes = []
    if not source.exists():
        return classes
    for entry in source.iterdir():
        if entry.is_dir() and entry.name not in IGNORED_SOURCE_DIRS and not entry.name.startswith('.'):
            classes.append(entry.name)
    classes.sort()
    return classes


def collect_pairs(class_dir: Path) -> List[str]:
    images_dir = class_dir / "images"
    labels_dir = class_dir / "labels"

    if not images_dir.exists() or not labels_dir.exists():
        return []

    # Only consider lowercase .jpg / .txt by spec. We'll normalize case for matching.
    image_basenames: Set[str] = set(
        p.stem for p in images_dir.glob("*.jpg") if p.is_file()
    )
    label_basenames: Set[str] = set(
        p.stem for p in labels_dir.glob("*.txt") if p.is_file()
    )

    paired = sorted(image_basenames & label_basenames)
    return paired


def split_indices(n: int, seed: int | None) -> Tuple[List[int], List[int], List[int]]:
    idx = list(range(n))
    rnd = random.Random(seed)
    rnd.shuffle(idx)
    n_train = int(0.80 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val
    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]
    assert len(test_idx) == n_test
    return train_idx, val_idx, test_idx


def copy_pair(source_class_dir: Path, target_split_class_dir: Path, basename: str) -> None:
    src_img = source_class_dir / "images" / f"{basename}.jpg"
    src_lbl = source_class_dir / "labels" / f"{basename}.txt"

    dst_img_dir = target_split_class_dir / "images"
    dst_lbl_dir = target_split_class_dir / "labels"
    ensure_dir(dst_img_dir)
    ensure_dir(dst_lbl_dir)

    shutil.copy2(src_img, dst_img_dir / src_img.name)
    shutil.copy2(src_lbl, dst_lbl_dir / src_lbl.name)


def build_target_structure(target: Path, classes: List[str]) -> None:
    for split in SPLITS:
        for cls in classes:
            ensure_dir(target / split / cls / "images")
            ensure_dir(target / split / cls / "labels")


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test with paired copies.")
    parser.add_argument("--source", required=True, type=Path, help="Path to the source root directory.")
    parser.add_argument("--target", required=True, type=Path, help="Path to the target root directory where files will be copied.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible shuffling (default: 42).")
    args = parser.parse_args(argv)

    source: Path = args.source
    target: Path = args.target
    seed: int = args.seed

    # Ensure source dir exists and create train/val/test under it (as per spec).
    ensure_dir(source)
    for split in SPLITS:
        ensure_dir(source / split)

    # Discover classes under source (excluding train/val/test).
    classes = find_classes(source)
    if not classes:
        print(f"No class directories found under {source}. Create class subdirectories with images/ and labels/.", file=sys.stderr)
        # Still ensure target root exists.
        ensure_dir(target)
        return 0

    # Prepare target directory structure
    build_target_structure(target, classes)

    total_pairs = 0
    summary: Dict[str, Tuple[int, int, int]] = {}

    for cls in classes:
        class_dir = source / cls
        basenames = collect_pairs(class_dir)
        n = len(basenames)
        if n == 0:
            print(f"[WARN] Class '{cls}' has no paired .jpg/.txt files; skipping.")
            summary[cls] = (0, 0, 0)
            continue

        train_idx, val_idx, test_idx = split_indices(n, seed)

        for i in train_idx:
            copy_pair(class_dir, target / "train" / cls, basenames[i])
        for i in val_idx:
            copy_pair(class_dir, target / "val" / cls, basenames[i])
        for i in test_idx:
            copy_pair(class_dir, target / "test" / cls, basenames[i])

        summary[cls] = (len(train_idx), len(val_idx), len(test_idx))
        total_pairs += n

    # Print summary
    print("Split summary (pairs copied):")
    for cls in classes:
        tr, va, te = summary.get(cls, (0, 0, 0))
        print(f"  {cls:20s} train={tr:4d}  val={va:4d}  test={te:4d}  total={tr+va+te:4d}")
    print(f"TOTAL pairs processed: {total_pairs}")
    print(f"Target written to: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
