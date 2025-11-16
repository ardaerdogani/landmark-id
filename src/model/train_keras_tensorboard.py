import json
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

from src.GLDV2_ds.gldv2_dataset import get_tf_datasets


CONFIG = {
    "arch": "MobileNetV3Small",
    "img_size": 224,
    "batch": 32,
    "epochs": 25,
    "optimizer": "AdamW",
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "augment": "flip,zoom,brightness,contrast",
    "seed": 42,
    "fine_tune_epochs": 10,
}


def _safe_override(env_key: str, cast_fn, target_key: str):
    raw_value = os.getenv(env_key)
    if raw_value is None:
        return
    try:
        CONFIG[target_key] = cast_fn(raw_value)
    except ValueError:
        print(f"Warning: Invalid value for {env_key}='{raw_value}', keeping {target_key}={CONFIG[target_key]}")


_safe_override("LANDMARK_EPOCHS", int, "epochs")
_safe_override("LANDMARK_FINE_TUNE_EPOCHS", lambda v: max(0, int(v)), "fine_tune_epochs")
_safe_override("LANDMARK_BATCH", lambda v: max(1, int(v)), "batch")
_safe_override("BATCH_SIZE", lambda v: max(1, int(v)), "batch")
_safe_override("LANDMARK_SHUFFLE_FRAC", float, "shuffle_frac")

CFG = SimpleNamespace(**CONFIG)

tf.keras.utils.set_random_seed(CFG.seed)

log_root = pathlib.Path(os.path.expanduser(os.getenv("LANDMARK_TB_ROOT", "tb_logs")))
timestamp = os.getenv("LANDMARK_TB_RUN") or datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = log_root / f"landmark_mnv3_{timestamp}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 60)
print("CONFIG")
print("=" * 60)
for k, v in CONFIG.items():
    print(f"{k}: {v}")
print(f"\n✓ TensorBoard logging: {LOG_DIR}")
print("=" * 60 + "\n")

IMG_SIZE = (CFG.img_size, CFG.img_size)

# Load streaming datasets
train_ds, class_names, train_samples = get_tf_datasets(
    "train", img_size=CFG.img_size, batch=CFG.batch, seed=CFG.seed
)
val_ds, _, val_samples = get_tf_datasets("val", img_size=CFG.img_size, batch=CFG.batch, seed=CFG.seed)
test_ds, _, test_samples = get_tf_datasets("test", img_size=CFG.img_size, batch=CFG.batch, seed=CFG.seed)

class_names = sorted(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

train_steps = max(1, math.ceil(train_samples / CFG.batch))
val_steps = max(1, math.ceil(val_samples / CFG.batch))
test_steps = max(1, math.ceil(test_samples / CFG.batch))
print(f"[train_keras_tensorboard] steps_per_epoch -> train={train_steps}, val={val_steps}, test={test_steps}")

# === class_weight (inverse-frequency) ===
metadata_file = pathlib.Path("data/metadata.json")
with open(metadata_file, "r") as f:
    metadata = json.load(f)
counts = {name: len(imgs) for name, imgs in metadata.items()}
total = sum(counts.values())
class_weight = {i: total / (len(class_names) * counts[name]) for i, name in enumerate(class_names)}

# ==== Augment ====
data_augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomZoom(0.12),
        layers.RandomBrightness(0.15),
        layers.RandomContrast(0.10),
    ],
    name="aug",
)

# ==== Model ====
base = keras.applications.MobileNetV3Small(
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

opt = keras.optimizers.AdamW(learning_rate=CFG.lr, weight_decay=CFG.weight_decay)
model.compile(
    optimizer=opt,
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("models", exist_ok=True)


def build_callbacks(prefix: str):
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=f"models/{prefix}ckpt_{{epoch:02d}}_{{val_loss:.3f}}.keras",
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
    return checkpoint, plateau, early_stop


tensorboard_cb = keras.callbacks.TensorBoard(
    log_dir=str(LOG_DIR),
    histogram_freq=1,
    profile_batch=0,
    write_graph=True,
    write_images=False,
)

phase1_ckpt, phase1_plateau, phase1_early = build_callbacks(prefix="phase1_")
phase1_callbacks = [tensorboard_cb, phase1_ckpt, phase1_plateau, phase1_early]

# ======= Training =======
print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60 + "\n")

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=CFG.epochs,
    callbacks=phase1_callbacks,
    verbose=1,
    class_weight=class_weight,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
)

# === Fine-tune stage ===
if CFG.fine_tune_epochs > 0:
    best_epoch_idx = None
    best_checkpoint_path = None
    if history and history.history.get("val_loss"):
        best_epoch_idx = int(np.argmin(history.history["val_loss"]))
        best_val_loss = history.history["val_loss"][best_epoch_idx]
        best_checkpoint_path = pathlib.Path("models") / f"phase1_ckpt_{best_epoch_idx + 1:02d}_{best_val_loss:.3f}.keras"
        if best_checkpoint_path.exists():
            model = tf.keras.models.load_model(best_checkpoint_path)
            print(f"Loaded weights from {best_checkpoint_path}")
        else:
            print(f"Warning: Expected checkpoint {best_checkpoint_path} not found; proceeding with current weights.")
    else:
        print("Warning: No validation loss history available; skipping checkpoint reload before fine-tune.")

    base.trainable = True

    for layer in base.layers:
        layer.trainable = False

    for layer in base.layers[-20:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    opt_ft = keras.optimizers.Adam(learning_rate=3e-5)

    model.compile(
        optimizer=opt_ft,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )

    phase2_ckpt, phase2_plateau, phase2_early = build_callbacks(prefix="phase2_")
    ft_callbacks = [tensorboard_cb, phase2_ckpt, phase2_plateau, phase2_early]

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CFG.fine_tune_epochs,
        callbacks=ft_callbacks,
        verbose=2,
        class_weight=class_weight,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
    )
else:
    print("Skipping fine-tune stage (fine_tune_epochs=0)")

# ======= Test Evaluation =======
test_metrics = model.evaluate(test_ds, steps=test_steps, verbose=0)
metrics_dict = {f"test_{name}": float(val) for name, val in zip(model.metrics_names, test_metrics)}
print(f"\nTest Metrics: {metrics_dict}")

# ======= Confusion Matrix =======
y_true, y_pred = [], []
test_ds_for_metrics, _, _ = get_tf_datasets("test", img_size=CFG.img_size, batch=CFG.batch, seed=CFG.seed)
test_ds_for_metrics = test_ds_for_metrics.prefetch(AUTOTUNE)

for bx, by in test_ds_for_metrics:
    preds = model.predict(bx, verbose=0)
    y_true.extend(np.argmax(by.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)

import matplotlib.pyplot as plt
import seaborn as sns

os.makedirs("reports", exist_ok=True)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Pred")
plt.ylabel("True")
plt.tight_layout()
cm_path = pathlib.Path("reports/confusion_matrix.png")
plt.savefig(cm_path, dpi=200)
plt.close()
print(f"Confusion matrix saved to {cm_path}")

# ======= Classification Report =======
report_text = classification_report(y_true, y_pred, target_names=class_names)
print("\n" + "=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)
print(report_text)

report_path = pathlib.Path("reports/classification_report.txt")
with open(report_path, "w") as f:
    f.write(report_text)
print(f"Classification report saved to {report_path}")


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

# ======= Save Model =======
model.save("models/landmark_mnv3.keras")
print("✓ Model saved to models/landmark_mnv3.keras")

print("Files saved:")
print("  - models/landmark_mnv3.keras (final model)")
print("  - models/ckpt_*.keras (best checkpoints)")
print("  - reports/confusion_matrix.png (evaluation)")
print("  - reports/classification_report.txt (evaluation)")
print("  - TensorBoard logs at", LOG_DIR)
print("=" * 60 + "\n")
