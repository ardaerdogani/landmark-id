# import argparse
# import json
# import shutil
# import sys
# from pathlib import Path
# from typing import Iterable, List

# import numpy as np
# import tensorflow as tf
# from tensorflow import keras

# ROOT = Path(__file__).resolve().parents[1]
# if str(ROOT) not in sys.path:
#     sys.path.insert(0, str(ROOT))

# from src.GLDV2_ds.gldv2_dataset import get_tf_datasets  # noqa: E402


# def build_inference_model(source_model: keras.Model, img_size: int) -> keras.Model:
#     """Create an inference-only graph without the data augmentation stack."""
#     inputs = keras.Input(shape=(img_size, img_size, 3), name="inference_input")
#     x = keras.applications.mobilenet_v3.preprocess_input(inputs)

#     backbone = source_model.get_layer("MobileNetV3Small")
#     gap = source_model.get_layer("global_average_pooling2d")
#     dropout = source_model.get_layer("dropout")
#     classifier = source_model.get_layer("dense")

#     x = backbone(x, training=False)
#     x = gap(x)
#     x = dropout(x, training=False)
#     outputs = classifier(x)

#     return keras.Model(inputs=inputs, outputs=outputs, name="landmark_mnv3_inference")


# def make_representative_dataset(dataset, limit: int) -> Iterable[List[np.ndarray]]:
#     """Yield batches for post-training quantization."""
#     count = 0
#     for images, _ in dataset:
#         yield [images.numpy().astype(np.float32)]
#         count += 1
#         if limit and count >= limit:
#             break


# def export_labels(metadata_path: Path, labels_path: Path) -> None:
#     labels = sorted(json.loads(Path(metadata_path).read_text()).keys())
#     labels_path.parent.mkdir(parents=True, exist_ok=True)
#     labels_path.write_text("\n".join(labels))


# def convert_models(
#     keras_model_path: Path,
#     saved_model_dir: Path,
#     fp32_path: Path,
#     int8_path: Path,
#     metadata_path: Path,
#     labels_path: Path,
#     img_size: int,
#     rep_samples: int,
#     skip_int8: bool,
# ) -> None:
#     print(f"Loading model from {keras_model_path} ...")
#     source_model = keras.models.load_model(keras_model_path, compile=False)
#     inference_model = build_inference_model(source_model, img_size)

#     if saved_model_dir.exists():
#         shutil.rmtree(saved_model_dir)
#     print(f"Exporting SavedModel to {saved_model_dir} ...")
#     inference_model.export(saved_model_dir)

#     print("Converting FP32 TFLite model ...")
#     converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
#     fp32_bytes = converter.convert()
#     fp32_path.parent.mkdir(parents=True, exist_ok=True)
#     fp32_path.write_bytes(fp32_bytes)
#     print(f"  -> wrote {fp32_path} ({len(fp32_bytes) / 1024:.1f} KB)")

#     if not skip_int8:
#         print("Preparing representative dataset ...")
#         rep_dataset, _, _ = get_tf_datasets("train", img_size=img_size, batch=1, seed=42)
#         rep_dataset = rep_dataset.take(rep_samples)

#         print("Converting INT8 TFLite model ...")
#         converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_dir))
#         converter.optimizations = [tf.lite.Optimize.DEFAULT]
#         converter.representative_dataset = lambda: make_representative_dataset(rep_dataset, rep_samples)
#         converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
#         converter.inference_input_type = tf.uint8
#         converter.inference_output_type = tf.uint8
#         int8_bytes = converter.convert()
#         int8_path.parent.mkdir(parents=True, exist_ok=True)
#         int8_path.write_bytes(int8_bytes)
#         print(f"  -> wrote {int8_path} ({len(int8_bytes) / 1024:.1f} KB)")
#     else:
#         print("Skipping INT8 conversion per flag.")

#     print("Writing label file ...")
#     export_labels(metadata_path, labels_path)
#     print(f"Labels saved to {labels_path}")
#     print("Export complete.")


# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Export Keras model to TFLite (FP32 + INT8).")
#     parser.add_argument("--keras-model", default="models/landmark_mnv3.keras", type=Path)
#     parser.add_argument("--saved-model-dir", default="models/export_savedmodel", type=Path)
#     parser.add_argument("--out-fp32", default="models/landmark_mnv3_fp32.tflite", type=Path)
#     parser.add_argument("--out-int8", default="models/landmark_mnv3_int8.tflite", type=Path)
#     parser.add_argument("--metadata", default="data/metadata.json", type=Path)
#     parser.add_argument("--labels-out", default="assets/labels.txt", type=Path)
#     parser.add_argument("--img-size", default=224, type=int)
#     parser.add_argument("--rep-samples", default=128, type=int, help="Representative batches for INT8 conversion.")
#     parser.add_argument("--skip-int8", action="store_true", help="Only create FP32 TFLite model.")
#     return parser.parse_args()


# def main() -> None:
#     args = parse_args()
#     convert_models(
#         keras_model_path=args.keras_model,
#         saved_model_dir=args.saved_model_dir,
#         fp32_path=args.out_fp32,
#         int8_path=args.out_int8,
#         metadata_path=args.metadata,
#         labels_path=args.labels_out,
#         img_size=args.img_size,
#         rep_samples=args.rep_samples,
#         skip_int8=args.skip_int8,
#     )


# if __name__ == "__main__":
#     main()
