import argparse
import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import tensorflow as tf
from tensorflow import keras


SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class Sample:
    path: str
    label_idx: int
    label_name: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Keras model on locally captured mobile photos."
    )
    parser.add_argument("--image-dir", required=True, type=Path, help="Root folder with <label>/image.jpg structure")
    parser.add_argument("--model", default=Path("models/landmark_mnv3.keras"), type=Path)
    parser.add_argument("--labels", default=Path("assets/labels.txt"), type=Path)
    parser.add_argument("--img-size", default=224, type=int)
    parser.add_argument("--batch", default=16, type=int)
    parser.add_argument("--topk", default=3, type=int)
    parser.add_argument(
        "--out",
        default=None,
        type=Path,
        help="Optional path for the JSON report (defaults to reports/mobile_eval_<ts>.json)",
    )
    return parser.parse_args()


def load_label_list(path: Path) -> List[str]:
    labels = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not labels:
        raise ValueError(f"No labels found in {path}")
    return labels


def gather_samples(root: Path, label_to_idx: Dict[str, int]) -> List[Sample]:
    samples: List[Sample] = []
    for class_dir in root.iterdir():
        if not class_dir.is_dir():
            continue
        class_name = class_dir.name
        if class_name not in label_to_idx:
            print(f"[eval_mobile_folder] Skipping '{class_name}' – not part of training labels")
            continue
        for file in class_dir.rglob("*"):
            if not file.is_file():
                continue
            if file.suffix.lower() not in SUPPORTED_SUFFIXES:
                continue
            samples.append(
                Sample(path=str(file), label_idx=label_to_idx[class_name], label_name=class_name)
            )
    return sorted(samples, key=lambda s: (s.label_idx, s.path))


def build_dataset(samples: Sequence[Sample], img_size: int, batch: int, num_classes: int) -> tf.data.Dataset:
    paths = tf.convert_to_tensor([s.path for s in samples], dtype=tf.string)
    labels = tf.convert_to_tensor([s.label_idx for s in samples], dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load(path, label):
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image = tf.image.resize(image, (img_size, img_size), method=tf.image.ResizeMethod.BILINEAR)
        image = tf.cast(image, tf.float32)
        label_one_hot = tf.one_hot(label, depth=num_classes)
        return image, label_one_hot

    return ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(batch).prefetch(tf.data.AUTOTUNE)


def main() -> None:
    args = parse_args()
    labels = load_label_list(args.labels)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    samples = gather_samples(args.image_dir, label_to_idx)
    if not samples:
        raise SystemExit(f"No images found under {args.image_dir}. Expected <label>/image.jpg folders.")

    dataset = build_dataset(samples, args.img_size, args.batch, num_classes=len(labels))
    model = keras.models.load_model(args.model)
    preds: List[np.ndarray] = []
    for batch_images, _ in dataset:
        batch_preds = model.predict(batch_images, verbose=0)
        preds.append(batch_preds)
    predictions = np.concatenate(preds, axis=0)
    true_labels = np.array([s.label_idx for s in samples])

    top1_idx = np.argmax(predictions, axis=1)
    top1_acc = float(np.mean(top1_idx == true_labels))

    topk = min(args.topk, predictions.shape[1])
    sorted_idx = np.argsort(predictions, axis=1)[:, -topk:][:, ::-1]
    topk_hits = np.any(sorted_idx == true_labels[:, None], axis=1)
    topk_acc = float(np.mean(topk_hits))

    per_class = {}
    for idx, label in enumerate(labels):
        mask = true_labels == idx
        if not np.any(mask):
            continue
        per_class[label] = float(np.mean(top1_idx[mask] == idx))

    rows = []
    for i, sample in enumerate(samples):
        ranking = sorted_idx[i]
        row = {
            "image": os.path.relpath(sample.path, args.image_dir),
            "true_label": sample.label_name,
            "topk": [
                {"label": labels[j], "score": float(predictions[i, j])}
                for j in ranking
            ],
            "correct_top1": bool(ranking[0] == sample.label_idx),
        }
        rows.append(row)

    report = {
        "image_root": str(args.image_dir),
        "model": str(args.model),
        "samples": len(samples),
        "top1_accuracy": top1_acc,
        f"top{topk}_accuracy": topk_acc,
        "per_class_top1": per_class,
        "predictions": rows,
    }

    out_path = args.out or Path("reports") / f"mobile_eval_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2))

    print(
        f"Mobile photos evaluated: {len(samples)} | Top-1={top1_acc:.3f} | Top-{topk}={topk_acc:.3f} -> {out_path}"
    )


if __name__ == "__main__":
    main()
