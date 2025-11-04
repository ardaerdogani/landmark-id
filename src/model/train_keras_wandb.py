import math
import os, pathlib, numpy as np, tensorflow as tf
import json
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import confusion_matrix, classification_report
from src.GLDV2_ds.gldv2_dataset import get_tf_datasets

# Import W&B for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import W&B: {e}")
    print("Continuing without W&B logging...")
    WANDB_AVAILABLE = False

# Config dict
CONFIG = {
    "arch":"MobileNetV3Small",
    "img_size":224,
    "batch":32,
    "epochs":25,
    "optimizer":"AdamW",
    "lr":1e-3,
    "weight_decay":1e-4,
    "augment":"flip,zoom,brightness,contrast",
    "seed":42,
    "fine_tune_epochs":10,
}

env_epochs = os.getenv("LANDMARK_EPOCHS")
if env_epochs:
    try:
        CONFIG["epochs"] = int(env_epochs)
    except ValueError:
        print(f"Warning: Invalid LANDMARK_EPOCHS value '{env_epochs}', falling back to default {CONFIG['epochs']}")

env_fine_tune = os.getenv("LANDMARK_FINE_TUNE_EPOCHS")
if env_fine_tune:
    try:
        CONFIG["fine_tune_epochs"] = max(0, int(env_fine_tune))
    except ValueError:
        print(f"Warning: Invalid LANDMARK_FINE_TUNE_EPOCHS value '{env_fine_tune}', using default {CONFIG['fine_tune_epochs']}")

env_batch = os.getenv("LANDMARK_BATCH") or os.getenv("BATCH_SIZE")
if env_batch:
    try:
        CONFIG["batch"] = max(1, int(env_batch))
    except ValueError:
        print(f"Warning: Invalid batch override value '{env_batch}', keeping default {CONFIG['batch']}")

env_shuffle = os.getenv("LANDMARK_SHUFFLE_FRAC")
if env_shuffle:
    try:
        CONFIG["shuffle_frac"] = float(env_shuffle)
    except ValueError:
        print(f"Warning: Invalid LANDMARK_SHUFFLE_FRAC value '{env_shuffle}', ignoring override.")

# Initialize W&B if available
if WANDB_AVAILABLE:
    try:
        wandb.init(
            project="landmark-id",
            name="mnv3small-gldv2",
            config=CONFIG,
            tags=["training", "mobilenetv3", "gld-v2"]
        )
        CFG = wandb.config
        print("✓ Connected to W&B!")
    except Exception as e:
        print(f"Warning: W&B init failed: {e}")
        print("Using local config instead...")
        class ConfigObj:
            def __init__(self, d):
                for k, v in d.items():
                    setattr(self, k, v)
        CFG = ConfigObj(CONFIG)
        WANDB_AVAILABLE = False
else:
    # Fallback: use config dict directly
    class ConfigObj:
        def __init__(self, d):
            for k, v in d.items():
                setattr(self, k, v)
    CFG = ConfigObj(CONFIG)

print("\n" + "="*60)
print("CONFIG")
print("="*60)
for k, v in CONFIG.items():
    print(f"{k}: {v}")
if WANDB_AVAILABLE:
    print("\n✓ W&B Logging: ENABLED")
else:
    print("\n✗ W&B Logging: DISABLED (local tracking only)")
print("="*60 + "\n")

tf.keras.utils.set_random_seed(CFG.seed)

IMG_SIZE = (CFG.img_size, CFG.img_size)
BATCH = CFG.batch

# Load streaming datasets
train_ds, class_names, train_samples = get_tf_datasets("train", img_size=CFG.img_size, batch=CFG.batch, seed=CFG.seed)
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
print(f"[train_keras_wandb] steps_per_epoch -> train={train_steps}, val={val_steps}, test={test_steps}")

# === class_weight (inverse-frequency) ===
metadata_file = pathlib.Path("data/metadata.json")
with open(metadata_file, 'r') as f:
    metadata = json.load(f)
counts = {name: len(imgs) for name, imgs in metadata.items()}
total = sum(counts.values())
class_weight = {i: total / (len(class_names) * counts[name]) for i, name in enumerate(class_names)}

# ==== Augment ====
data_augment = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomZoom(0.12),
    layers.RandomBrightness(0.15),
    layers.RandomContrast(0.10),
], name="aug")

# ==== Model ====
base = keras.applications.MobileNetV3Small(
    input_shape=IMG_SIZE + (3,), include_top=False, weights="imagenet")
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

# ==== Callback Helpers ====
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


phase1_ckpt, phase1_plateau, phase1_early = build_callbacks(prefix="phase1_")
phase1_callbacks = [phase1_ckpt, phase1_plateau, phase1_early]

# ======= Training =======
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

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

# === Fine-tune (daha az layer + BN freeze) ===
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

    # Önce tüm katmanları kapat (BatchNorm dahil)
    for layer in base.layers:
        layer.trainable = False

    # BN katmanlarını dondur, diğerlerinden son 20 tanesini aç
    for layer in base.layers[-20:]:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True

    # Yeni optimizer ile düşük LR (decay kapalı)
    opt_ft = keras.optimizers.Adam(learning_rate=3e-5)

    # Label smoothing ile compile
    model.compile(
        optimizer=opt_ft,
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.05),
        metrics=["accuracy"],
    )
    # Kısa fine-tune (+ ReduceLROnPlateau)
    phase2_ckpt, phase2_plateau, phase2_early = build_callbacks(prefix="phase2_")
    ft_callbacks = [phase2_ckpt, phase2_plateau, phase2_early]

    ft_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=CFG.fine_tune_epochs,
        callbacks=ft_callbacks,
        verbose=2,
        class_weight=class_weight,  # <-- eklendi
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
    )
else:
    print("Skipping fine-tune stage (fine_tune_epochs=0)")

# ======= Test Evaluation =======
test_metrics = model.evaluate(test_ds, steps=test_steps, verbose=0)
metrics_dict = {f"test_{name}": float(val) for name, val in zip(model.metrics_names, test_metrics)}
print(f"\nTest Metrics: {metrics_dict}")

if WANDB_AVAILABLE:
    try:
        wandb.log(metrics_dict)
    except Exception as e:
        print(f"Warning: Could not log metrics to W&B: {e}")

# ======= Confusion Matrix =======
y_true, y_pred = [], []
# Reload test dataset for evaluation loops to ensure fresh iterator
test_ds_for_metrics, _, _ = get_tf_datasets("test", img_size=CFG.img_size, batch=CFG.batch, seed=CFG.seed)
test_ds_for_metrics = test_ds_for_metrics.prefetch(AUTOTUNE)

for bx, by in test_ds_for_metrics:
    preds = model.predict(bx, verbose=0)
    y_true.extend(np.argmax(by.numpy(), axis=1))
    y_pred.extend(np.argmax(preds, axis=1))
cm = confusion_matrix(y_true, y_pred)

# Save confusion matrix visualization
import matplotlib.pyplot as plt
import seaborn as sns
os.makedirs("reports", exist_ok=True)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Pred"); plt.ylabel("True"); plt.tight_layout()
plt.savefig("reports/confusion_matrix.png", dpi=200)
print("Confusion matrix saved to reports/confusion_matrix.png")

# Log confusion matrix to W&B if available
if WANDB_AVAILABLE:
    try:
        wandb.log({"confusion_matrix": wandb.Image("reports/confusion_matrix.png")})
    except Exception as e:
        print(f"Warning: Could not log confusion matrix to W&B: {e}")

# ======= Classification Report =======
report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
print("\n" + "="*60)
print("CLASSIFICATION REPORT")
print("="*60)
print(classification_report(y_true, y_pred, target_names=class_names))

# ======= Save Model =======
model.save("models/landmark_mnv3.keras")
print("✓ Model saved to models/landmark_mnv3.keras")

# Log model artifact to W&B if available
if WANDB_AVAILABLE:
    try:
        artifact = wandb.Artifact("landmark-mnv3", type="model")
        artifact.add_file("models/landmark_mnv3.keras")
        wandb.log_artifact(artifact)
        print("✓ Model artifact logged to W&B")
    except Exception as e:
        print(f"Warning: Could not log artifact to W&B: {e}")


print("Files saved:")
print("  - models/landmark_mnv3.keras (final model)")
print("  - models/ckpt_*.keras (best checkpoints)")
print("  - reports/confusion_matrix.png (evaluation)")
print("="*60 + "\n")

# Finish W&B run if available
if WANDB_AVAILABLE:
    try:
        wandb.finish()
        print("W&B run finalized")
    except Exception as e:
        print(f"Warning: Could not finalize W&B run: {e}")
