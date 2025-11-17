"""Utility helpers for mixing locally captured mobile photos into training/eval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import tensorflow as tf

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _gather_samples(root: Path, class_names: List[str]) -> Tuple[List[str], List[int], List[str]]:
    file_paths: List[str] = []
    labels: List[int] = []
    missing: List[str] = []

    for idx, cls in enumerate(class_names):
        cls_dir = root / cls
        if not cls_dir.exists():
            missing.append(cls)
            continue

        for path in cls_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            file_paths.append(str(path))
            labels.append(idx)

    return file_paths, labels, missing


def _make_dataset(
    paths: Iterable[str],
    labels: Iterable[int],
    img_size: int,
    batch: int,
    seed: int,
    num_classes: int,
) -> tf.data.Dataset:
    path_list = list(paths)
    label_list = list(labels)
    path_tensor = tf.convert_to_tensor(path_list, dtype=tf.string)
    label_tensor = tf.convert_to_tensor(label_list, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((path_tensor, label_tensor))
    buffer = max(len(path_list), batch * 2, 32)
    ds = ds.shuffle(buffer_size=buffer, seed=seed, reshuffle_each_iteration=True)

    def _load_image(path: tf.Tensor, label: tf.Tensor):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.float32)
        label_one_hot = tf.one_hot(label, depth=num_classes)
        return image, label_one_hot

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.repeat()
    ds = ds.batch(batch, drop_remainder=False)
    return ds


def build_mobile_dataset(
    root_dir: os.PathLike | str,
    class_names: List[str],
    img_size: int,
    batch: int,
    seed: int,
) -> Tuple[Optional[tf.data.Dataset], int]:
    """Return a tf.data.Dataset built from `root_dir/<class>/*.jpg` photos."""

    root = Path(root_dir)
    if not root.exists():
        print(f"[mobile_folder_dataset] '{root}' does not exist; skipping mobile mixing")
        return None, 0

    paths, labels, missing = _gather_samples(root, class_names)
    if missing:
        print(
            "[mobile_folder_dataset] Missing class folders for:",
            ", ".join(sorted(missing)),
        )

    if not paths:
        print(
            f"[mobile_folder_dataset] No compatible images found under {root}; expected a folder per class name."
        )
        return None, 0

    dataset = _make_dataset(paths, labels, img_size, batch, seed, num_classes=len(class_names))
    print(
        f"[mobile_folder_dataset] Loaded {len(paths)} mobile photos from {root} (classes present: {len(set(labels))})"
    )
    return dataset, len(paths)
