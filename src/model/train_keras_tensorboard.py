import math
import os
import pathlib
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers


# =========================
# Config (override via env)
# =========================
CONFIG = {
    "arch": "MobileNetV3Large",
    "img_size": 224,
    "batch": 32,
    "epochs": 10,
    "optimizer": "AdamW",
    "lr": 3e-4,
    "weight_decay": 1e-4,
    "seed": 42,
    "fine_tune_epochs": 10,
    "fine_tune_unfreeze_layers": 60,
    "fine_tune_lr": 1e-5,
    "use_class_weight": True,
    "label_smoothing": 0.0,
}


def _safe_override(env_key: str, cast_fn, target_key: str):
    raw_value = os.getenv(env_key)
    if raw_value is None:
        return
    try:
        CONFIG[target_key] = cast_fn(raw_value)
    except ValueError:
        print(f"Warning: Invalid value for {env_key}='{raw_value}', keeping {target_key}={CONFIG[target_key]}")


_safe_override("LANDMARK_IMG_SIZE", int, "img_size")
_safe_override("LANDMARK_EPOCHS", int, "epochs")
_safe_override("LANDMARK_FINE_TUNE_EPOCHS", lambda v: max(0, int(v)), "fine_tune_epochs")
_safe_override("LANDMARK_BATCH", lambda v: max(1, int(v)), "batch")
_safe_override("BATCH_SIZE", lambda v: max(1, int(v)), "batch")
_safe_override("LANDMARK_LR", float, "lr")
_safe_override("LANDMARK_WEIGHT_DECAY", float, "weight_decay")
_safe_override("LANDMARK_FINE_TUNE_UNFREEZE", lambda v: int(v), "fine_tune_unfreeze_layers")
_safe_override("LANDMARK_FINE_TUNE_LR", float, "fine_tune_lr")
_safe_override("LANDMARK_USE_CLASS_WEIGHT", lambda v: str(v).strip().lower() in ("1", "true", "yes", "y"), "use_class_weight")
_safe_override("LANDMARK_LABEL_SMOOTHING", float, "label_smoothing")

CFG = SimpleNamespace(**CONFIG)

tf.keras.utils.set_random_seed(CFG.seed)

# -------------------------
# Paths
# -------------------------
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DATA_ROOT = pathlib.Path(os.getenv("LANDMARK_DATA_ROOT", str(REPO_ROOT / "src" / "roboflow_dataset")))
LABELS_PATH = pathlib.Path(os.getenv("LANDMARK_LABELS_PATH", str(DATA_ROOT / "labels.txt")))
MODELS_DIR = pathlib.Path(os.getenv("LANDMARK_MODELS_DIR", str(REPO_ROOT / "models")))
REPORTS_DIR = pathlib.Path(os.getenv("LANDMARK_REPORTS_DIR", str(REPO_ROOT / "reports")))

MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# TensorBoard
log_root = pathlib.Path(os.path.expanduser(os.getenv("LANDMARK_TB_ROOT", "tb_logs")))
timestamp = os.getenv("LANDMARK_TB_RUN") or datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = log_root / f"landmark_mnv3_{timestamp}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 70)
print("TRAIN CONFIG")
print("=" * 70)
for k, v in CONFIG.items():
    print(f"{k}: {v}")
print(f"DATA_ROOT: {DATA_ROOT}")
print(f"LABELS_PATH: {LABELS_PATH}")
print(f"MODELS_DIR: {MODELS_DIR}")
print(f"REPORTS_DIR: {REPORTS_DIR}")
print(f"✓ TensorBoard logging: {LOG_DIR}")
print("=" * 70 + "\n")


# =========================
# Helpers
# =========================
IMG_SIZE = (CFG.img_size, CFG.img_size)
AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _read_labels(p: pathlib.Path) -> list[str]:
    if not p.exists():
        return []
    lines = []
    for ln in p.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        lines.append(ln)
    return lines


def _infer_labels_from_train(train_dir: pathlib.Path) -> list[str]:
    return sorted([d.name for d in train_dir.iterdir() if d.is_dir()])


def _count_images(dir_path: pathlib.Path) -> int:
    if not dir_path.exists():
        return 0
    return sum(1 for f in dir_path.rglob("*") if f.is_file() and f.suffix.lower() in IMAGE_EXTS)


def _resolve_split_dirs(root: pathlib.Path) -> tuple[pathlib.Path, pathlib.Path, pathlib.Path]:
    train_dir = root / "train"
    # Roboflow uses "valid" (not "val")
    val_dir = root / "valid" if (root / "valid").exists() else (root / "val")
    test_dir = root / "test"
    if not train_dir.exists():
        raise SystemExit(f"Missing train dir: {train_dir}")
    if not val_dir.exists():
        raise SystemExit(f"Missing validation dir: {val_dir} (expected 'valid' or 'val')")
    if not test_dir.exists():
        raise SystemExit(f"Missing test dir: {test_dir}")
    return train_dir, val_dir, test_dir


# =========================
# Load datasets (Roboflow)
# =========================
train_dir, val_dir, test_dir = _resolve_split_dirs(DATA_ROOT)

class_names = _read_labels(LABELS_PATH)
if not class_names:
    class_names = _infer_labels_from_train(train_dir)
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text("\n".join(class_names) + "\n", encoding="utf-8")
    print(f"[train] Wrote inferred labels to {LABELS_PATH}")

print(f"[train] num_classes={len(class_names)}")
if len(class_names) < 2:
    raise SystemExit("Need at least 2 classes to train.")

# IMPORTANT: we pass class_names=... to lock the index order
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=CFG.batch,
    shuffle=True,
    seed=CFG.seed,
)
val_ds = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=CFG.batch,
    shuffle=False,
)
test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels="inferred",
    label_mode="categorical",
    class_names=class_names,
    image_size=IMG_SIZE,
    batch_size=CFG.batch,
    shuffle=False,
)

train_samples = _count_images(train_dir)
val_samples = _count_images(val_dir)
test_samples = _count_images(test_dir)

train_steps = max(1, math.ceil(train_samples / CFG.batch))
val_steps = max(1, math.ceil(val_samples / CFG.batch))
test_steps = max(1, math.ceil(test_samples / CFG.batch))

print(f"[train] samples -> train={train_samples}, val={val_samples}, test={test_samples}")
print(f"[train] steps_per_epoch -> train={train_steps}, val={val_steps}, test={test_steps}")

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# =========================
# class_weight (optional)
# =========================
class_weight = None
if CFG.use_class_weight:
    counts = {name: _count_images(train_dir / name) for name in class_names}
    total = sum(counts.values())
    # inverse-frequency; guard against zero
    class_weight = {
        i: (total / (len(class_names) * max(1, counts[name])))
        for i, name in enumerate(class_names)
    }
    print("[train] class counts:", counts)


# =========================
# Augment + Model
# =========================
data_augment = keras.Sequential(
    [
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.1),
    ],
    name="aug",
)

base = keras.applications.MobileNetV3Large(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet"
)
base.trainable = False

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augment(inputs)
x = keras.applications.mobilenet_v3.preprocess_input(x)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(class_names), activation="softmax")(x)
model = keras.Model(inputs, outputs)

if str(CFG.optimizer).lower() == "adamw":
    opt = keras.optimizers.AdamW(learning_rate=CFG.lr, weight_decay=CFG.weight_decay)
else:
    opt = keras.optimizers.Adam(learning_rate=CFG.lr)

model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=[
        keras.metrics.CategoricalAccuracy(name="top1"),
        keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
    ],
)

# =========================
# Callbacks
# =========================
tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=str(LOG_DIR),
    histogram_freq=1,
    profile_batch=0,
    write_graph=True,
    write_images=False,
)

phase1_best = MODELS_DIR / "phase1_best.keras"
phase2_best = MODELS_DIR / "phase2_best.keras"

phase1_ckpt = keras.callbacks.ModelCheckpoint(
    filepath=str(phase1_best),
    monitor="val_loss",
    mode="min",
    save_best_only=True,
    save_weights_only=False,
)

plateau = keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    mode="min",
    factor=0.2,
    patience=2,
    min_lr=1e-7,
    verbose=1,
)

early_stop = keras.callbacks.EarlyStopping(
    monitor="val_loss",
    mode="min",
    patience=4,
    restore_best_weights=True,
    verbose=1,
)

phase1_callbacks = [tensorboard_cb, phase1_ckpt, plateau, early_stop]

print("\n" + "=" * 70)
print("STARTING TRAINING (phase 1: frozen backbone)")
print("=" * 70 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG.epochs,
    callbacks=phase1_callbacks,
    verbose=1,
    class_weight=class_weight,
)

# Reload best phase1 model if available
if phase1_best.exists():
    model = tf.keras.models.load_model(phase1_best)
    print(f"[train] Loaded best phase1 model from {phase1_best}")

# =========================
# Fine-tune
# =========================
if CFG.fine_tune_epochs > 0:
    print("\n" + "=" * 70)
    print("FINE-TUNING (phase 2: unfreeze last layers)")
    print("=" * 70 + "\n")

    base.trainable = True

    # freeze everything first
    for layer in base.layers:
        layer.trainable = False

    # unfreeze last N (except BatchNorm)
    layers_to_unfreeze = len(base.layers) if CFG.fine_tune_unfreeze_layers <= 0 else min(len(base.layers), CFG.fine_tune_unfreeze_layers)
    for layer in base.layers[-layers_to_unfreeze:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=CFG.fine_tune_lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=float(CFG.label_smoothing)),
        metrics=[
            keras.metrics.CategoricalAccuracy(name="top1"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
        ],
    )

    phase2_ckpt = keras.callbacks.ModelCheckpoint(
        filepath=str(phase2_best),
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        save_weights_only=False,
    )
    ft_callbacks = [tensorboard_cb, phase2_ckpt, plateau, early_stop]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CFG.fine_tune_epochs,
        callbacks=ft_callbacks,
        verbose=2,
        class_weight=class_weight
    )

    if phase2_best.exists():
        model = tf.keras.models.load_model(phase2_best)
        print(f"[train] Loaded best phase2 model from {phase2_best}")
else:
    print("[train] Skipping fine-tune stage (fine_tune_epochs=0)")

# =========================
# Evaluate on test
# =========================
test_metrics = model.evaluate(test_ds, verbose=0)
metrics_dict = {f"test_{name}": float(val) for name, val in zip(model.metrics_names, test_metrics)}
print(f"\nTest Metrics: {metrics_dict}")

# Confusion matrix + report
y_true, y_pred = [], []
for bx, by in test_ds:
    preds = model.predict(bx, verbose=0)
    y_true.extend(np.argmax(by.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))

cm = confusion_matrix(y_true, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Pred")
plt.ylabel("True")
plt.tight_layout()

cm_path = REPORTS_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"Confusion matrix saved to {cm_path}")

report_text = classification_report(y_true, y_pred, target_names=class_names, digits=4)
print("\n" + "=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)
print(report_text)

report_path = REPORTS_DIR / "classification_report.txt"
report_path.write_text(report_text, encoding="utf-8")
print(f"Classification report saved to {report_path}")

# Log eval to TensorBoard
def log_evaluation_to_tensorboard(metrics, confusion_matrix_path, report_str):
    eval_dir = LOG_DIR / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)
    writer = tf.summary.create_file_writer(str(eval_dir))
    with writer.as_default():
        for key, value in metrics.items():
            tf.summary.scalar(key, value, step=0)
        tf.summary.text("classification_report", tf.constant(report_str), step=0)
        if confusion_matrix_path.exists():
            image = tf.io.read_file(str(confusion_matrix_path))
            image = tf.image.decode_png(image, channels=4)
            image = tf.cast(tf.expand_dims(image, axis=0), tf.float32) / 255.0
            tf.summary.image("confusion_matrix", image, step=0)
    writer.flush()


log_evaluation_to_tensorboard(metrics_dict, cm_path, report_text)
print(f"TensorBoard evaluation summaries written to {LOG_DIR / 'evaluation'}")

# =========================
# Save final model + labels
# =========================
final_model_path = MODELS_DIR / "landmark_mnv3.keras"
model.save(final_model_path)
print(f"✓ Model saved to {final_model_path}")

# Save labels used for training in the same order as model outputs
labels_out = MODELS_DIR / "labels.txt"
labels_out.write_text("\n".join(class_names) + "\n", encoding="utf-8")
print(f"✓ Labels saved to {labels_out}")

print("\nFiles saved:")
print(f"  - {final_model_path}")
print(f"  - {phase1_best} (best phase1)")
print(f"  - {phase2_best} (best phase2, if fine-tuned)")
print(f"  - {labels_out}")
print(f"  - {cm_path}")
print(f"  - {report_path}")
print("  - TensorBoard logs at", LOG_DIR)