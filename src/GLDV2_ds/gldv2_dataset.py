
import json
import math
import os
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
import requests

PROJECT_ROOT = Path(__file__).parent.parent.parent
CACHE_ROOT = PROJECT_ROOT / "data" / "image_cache"
_PREPARED_SPLITS: Dict[str, List[Tuple[str, str]]] = {}
_SKIPPED_SPLITS: Dict[str, int] = {}


# Load metadata.json with all image URLs
@lru_cache(maxsize=1)
def load_metadata() -> Dict[str, List[Dict]]:
    metadata_file = PROJECT_ROOT / "data" / "metadata.json"
    with open(metadata_file, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_split(split_name: str) -> List[Tuple[str, str]]:
    split_file = PROJECT_ROOT / "data" / f"{split_name}.txt"
    data = []
    with open(split_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.rsplit(',', 1)
                if len(parts) == 2:
                    data.append((parts[0], parts[1]))
    return data


# Create mapping from landmark_name to class_id
def get_landmark_to_class_id(metadata: Dict[str, List[Dict]]) -> Dict[str, int]:
    mapping = {}
    for class_id, landmark_name in enumerate(sorted(metadata.keys())):
        mapping[landmark_name] = class_id
    return mapping


@lru_cache(maxsize=1)
def _metadata_index() -> Dict[str, Dict[str, Dict]]:
    metadata = load_metadata()
    index: Dict[str, Dict[str, Dict]] = {}
    for landmark_name, records in metadata.items():
        index[landmark_name] = {}
        for record in records:
            image_id = record.get('image_id')
            if image_id:
                index[landmark_name][image_id] = record
    return index


def _prepare_split(split_name: str, split_data: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
    if split_name in _PREPARED_SPLITS:
        return _PREPARED_SPLITS[split_name]

    metadata_index = _metadata_index()
    valid_entries: List[Tuple[str, str]] = []
    skipped = 0

    for landmark_name, image_id in split_data:
        landmark_records = metadata_index.get(landmark_name)
        if not landmark_records:
            skipped += 1
            continue

        image_entry = landmark_records.get(image_id)
        if not image_entry:
            skipped += 1
            continue

        url = image_entry.get('url')
        if not url:
            skipped += 1
            continue

        cache_key = f"{image_id}"
        img_array = download_image(url, cache_key=cache_key, split_name=split_name)
        if img_array is None:
            skipped += 1
            continue

        valid_entries.append((landmark_name, image_id))

    _PREPARED_SPLITS[split_name] = valid_entries
    _SKIPPED_SPLITS[split_name] = skipped

    if skipped:
        print(f"[gldv2_dataset] {split_name}: skipped {skipped} entries with missing images")

    return valid_entries


# Download image from URL (with simple disk cache)
def download_image(url: str, cache_key: str, split_name: str, timeout: int = 10) -> Optional[np.ndarray]:
    cache_dir = CACHE_ROOT / split_name
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{cache_key}.jpg"

    if cache_path.exists():
        try:
            with Image.open(cache_path) as cached_img:
                return np.array(cached_img.convert('RGB'))
        except Exception:
            # Corrupted cache entry -> remove and fallback to fresh download
            try:
                cache_path.unlink()
            except FileNotFoundError:
                pass

    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, timeout=timeout, headers=headers)
        response.raise_for_status()
        img = Image.open(BytesIO(response.content)).convert('RGB')
        # Persist to cache for future epochs
        try:
            img.save(cache_path, format='JPEG', quality=95)
        except Exception:
            pass
        return np.array(img)
    except Exception:
        return None


# Generator (image, label) pairs
def create_image_generator(split_name: str, metadata_index: Dict[str, Dict[str, Dict]],
                           landmark_to_class: Dict[str, int], img_size: int = 224,
                           split_entries: Optional[List[Tuple[str, str]]] = None):
    if split_entries is None:
        split_entries = _prepare_split(split_name, load_split(split_name))
    
    for landmark_name, image_id in split_entries:
        image_entry = metadata_index.get(landmark_name, {}).get(image_id)
        if not image_entry:
            continue
        
        url = image_entry.get('url')
        if not url:
            continue
        
        # Download and process image
        cache_key = f"{image_id}"
        img_array = download_image(url, cache_key=cache_key, split_name=split_name)
        if img_array is None:
            continue
        
        # Resize to img_size
        img_pil = Image.fromarray(img_array)
        img_pil = img_pil.resize((img_size, img_size), Image.BILINEAR)
        img_array = np.array(img_pil, dtype=np.float32)
        
        # Get class label
        class_id = landmark_to_class[landmark_name]
        label = tf.one_hot(class_id, depth=len(landmark_to_class))
        
        yield img_array, label


# Create TensorFlow datasets for train or validation
def get_tf_datasets(split_name: str = "train", img_size: int = 224, batch: int = 32, seed: int = 42):
    metadata = load_metadata()
    metadata_index = _metadata_index()
    landmark_to_class = get_landmark_to_class_id(metadata)
    split_data = load_split(split_name)
    split_entries = _prepare_split(split_name, split_data)
    num_classes = len(landmark_to_class)
    sample_count = len(split_entries)
    shuffle_buffer = None
    shuffle_frac = None
    shuffle_env = os.getenv("LANDMARK_SHUFFLE_FRAC")
    if shuffle_env:
        try:
            shuffle_frac = float(shuffle_env)
            if shuffle_frac <= 0:
                raise ValueError
        except ValueError:
            print(f"Warning: Invalid LANDMARK_SHUFFLE_FRAC value '{shuffle_env}', using default full-buffer shuffling.")
            shuffle_frac = None
    
    ds = tf.data.Dataset.from_generator(
        lambda: create_image_generator(split_name, metadata_index, landmark_to_class, img_size, split_entries=split_entries),
        output_signature=(
            tf.TensorSpec(shape=(img_size, img_size, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(num_classes,), dtype=tf.float32)
        )
    )
    
    if split_name == "train":
        if shuffle_frac is not None:
            shuffle_buffer = max(int(sample_count * shuffle_frac), batch, 1)
        else:
            shuffle_buffer = sample_count
        shuffle_buffer = max(shuffle_buffer, batch * 2, 32)
        ds = ds.shuffle(buffer_size=shuffle_buffer, seed=seed, reshuffle_each_iteration=True)
        ds = ds.repeat()
    
    ds = ds.batch(batch, drop_remainder=False)
    approx_steps = max(1, math.ceil(sample_count / batch))
    buffer_msg = f", shuffle_buffer={shuffle_buffer}" if shuffle_buffer is not None else ""
    dropped = _SKIPPED_SPLITS.get(split_name, 0)
    dropped_msg = f", dropped={dropped}" if dropped else ""
    print(f"[gldv2_dataset] split={split_name} samples={sample_count} batch={batch} steps≈{approx_steps}{buffer_msg}{dropped_msg}")
    
    return ds, list(landmark_to_class.keys()), sample_count


# Create both train and val TensorFlow datasets
def get_tf_datasets_pair(img_size: int = 224, batch: int = 32, seed: int = 42):
    train_ds, class_names, _ = get_tf_datasets("train", img_size, batch, seed)
    val_ds, _, _ = get_tf_datasets("val", img_size, batch, seed)
    
    return train_ds, val_ds
