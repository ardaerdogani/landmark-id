import math
import os, numpy as np, tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from pathlib import Path

from src.GLDV2_ds.gldv2_dataset import get_tf_datasets

IMG=(224,224); SEED=42; BATCH=32
test_ds, class_names, sample_count = get_tf_datasets("test", img_size=IMG[0], batch=BATCH, seed=SEED)
classes = sorted(class_names)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)
test_steps = max(1, math.ceil(sample_count / BATCH))

model = keras.models.load_model("models/landmark_mnv3.keras")
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=[
        keras.metrics.TopKCategoricalAccuracy(k=1, name="top1"),
        keras.metrics.TopKCategoricalAccuracy(k=3, name="top3"),
    ],
)
res = dict(zip(model.metrics_names, model.evaluate(test_ds, steps=test_steps, verbose=0)))

# Reload dataset for detailed metrics to ensure fresh iterator
test_ds, _, _ = get_tf_datasets("test", img_size=IMG[0], batch=BATCH, seed=SEED)
test_ds = test_ds.prefetch(tf.data.AUTOTUNE)

y_true,y_pred=[],[]
for bx,by in test_ds:
  p = model.predict(bx, verbose=0)
  y_true += list(np.argmax(by.numpy(),1))
  y_pred += list(np.argmax(p,1))

cm = confusion_matrix(y_true,y_pred)
rep = classification_report(y_true,y_pred,target_names=classes,output_dict=True)

Path("reports").mkdir(parents=True, exist_ok=True)
np.savetxt("reports/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
with open("reports/sprint1_metrics.md","w",encoding="utf-8") as f:
  f.write("# Sprint 1 – Metrics (Vilnius Commons)\n\n")
  f.write(f"- Test Top-1: **{res['top1']:.3f}**\n- Test Top-3: **{res['top3']:.3f}**\n")
  f.write(f"- Samples: **{len(y_true)}**, Classes: **{len(classes)}**\n\n")
  f.write("## Class-wise\n")
  for c in classes:
    d=rep[c]; f.write(f"- **{c}**: P={d['precision']:.3f}, R={d['recall']:.3f}, F1={d['f1-score']:.3f}\n")
print("Rapor hazır -> reports/sprint1_metrics.md + confusion_matrix.csv")
