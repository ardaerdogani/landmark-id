#!/usr/bin/env python3
"""Quick sanity check that a TFLite model's output dimension matches a labels file."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tensorflow as tf


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate TFLite model output vs labels length.")
    parser.add_argument("--model", default="exports/landmark_mnv3_int8_drq.tflite", help="Path to .tflite file")
    parser.add_argument("--labels", default="backend/LandmarkApi/Models/labels.txt", help="Path to labels.txt")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    model_path = Path(args.model)
    labels_path = Path(args.labels)

    if not model_path.exists():
        sys.stderr.write(f"Model not found: {model_path}\n")
        return 1
    if not labels_path.exists():
        sys.stderr.write(f"Labels file not found: {labels_path}\n")
        return 1

    labels = [ln.strip() for ln in labels_path.read_text().splitlines() if ln.strip()]

    interpreter = tf.lite.Interpreter(model_path=str(model_path))
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    out_dim = int(output_details["shape"][-1])

    print("Model:", model_path)
    print("Labels:", labels_path)
    print(f"Input shape: {input_details['shape']}, dtype={input_details['dtype']}")
    print(f"Output shape: {output_details['shape']}, dtype={output_details['dtype']}")
    print(f"Labels count: {len(labels)}")

    if out_dim != len(labels):
        sys.stderr.write(
            f"❌ Mismatch: model outputs {out_dim} classes but labels.txt has {len(labels)} entries\n"
        )
        return 2

    print("✅ Model output dimension matches labels length.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
