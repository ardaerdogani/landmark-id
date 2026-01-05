import tensorflow as tf
from typing import Tuple

def load_autosplit_datasets(
    root_dir: str,
    img_size: int,
    batch_size: int,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, list]:
    """
    Load images from root_dir/<class>/* and split into train/val/test in-memory.
    """

    assert train_ratio + val_ratio < 1.0, "train_ratio + val_ratio must be < 1"

    # Load everything first (no split)
    full_ds = tf.keras.utils.image_dataset_from_directory(
        root_dir,
        image_size=(img_size, img_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        label_mode="categorical",
    )

    class_names = full_ds.class_names
    num_samples = tf.data.experimental.cardinality(full_ds).numpy()

    # Convert batches → individual samples
    full_ds = full_ds.unbatch()

    # Shuffle once deterministically
    full_ds = full_ds.shuffle(
        buffer_size=10_000,
        seed=seed,
        reshuffle_each_iteration=False,
    )

    train_count = int(train_ratio * num_samples)
    val_count = int(val_ratio * num_samples)

    train_ds = full_ds.take(train_count)
    val_ds = full_ds.skip(train_count).take(val_count)
    test_ds = full_ds.skip(train_count + val_count)

    # Re-batch
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    print(
        f"[autosplit] samples={num_samples} "
        f"train={train_count} "
        f"val={val_count} "
        f"test={num_samples - train_count - val_count}"
    )

    return train_ds, val_ds, test_ds, class_names
