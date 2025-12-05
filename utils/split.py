import os
import shutil
import random
from pathlib import Path

def split_train_valid(base_dir: str, train_prefix="train", valid_prefix="valid", split_ratio=0.8, seed=42):
    """
    Splits dataset patches into train/valid sets across multiple modalities.
    Assumes filenames are prefixed by channel (e.g., nir_001.tif, red_001.tif).
    
    Parameters
    ----------
    base_dir : str
        Path to dataset directory containing folders like train_nir, train_red, etc.
    train_prefix : str
        Prefix for training folders (default: 'train').
    valid_prefix : str
        Prefix for validation folders (default: 'valid').
    split_ratio : float
        Proportion of data to use for training (default: 0.8).
    seed : int
        Random seed for reproducibility (default: 42).
    """

    random.seed(seed)

    # Modalities and their filename prefixes
    modalities = {
        "nir": "nir_",
        "blue": "blue_",
        "green": "green_",
        "red": "red_",
        "gt": "gt_",
    }

    # Ensure validation directories exist
    for m in modalities:
        valid_dir = Path(base_dir) / f"{valid_prefix}_{m}"
        valid_dir.mkdir(parents=True, exist_ok=True)

    # Use filenames from one modality (nir) to determine patch IDs
    nir_dir = Path(base_dir) / f"{train_prefix}_nir"
    filenames = sorted([f for f in os.listdir(nir_dir) if os.path.isfile(nir_dir / f)])

    # Extract patch IDs (remove prefix and extension)
    patch_ids = [Path(f).stem.replace(modalities["nir"], "") for f in filenames]

    # Shuffle and split
    random.shuffle(patch_ids)
    split_idx = int(len(patch_ids) * split_ratio)
    train_ids = patch_ids[:split_idx]
    valid_ids = patch_ids[split_idx:]

    print(f"Total patches: {len(patch_ids)} | Train: {len(train_ids)} | Valid: {len(valid_ids)}")

    # Move validation files across all modalities
    for pid in valid_ids:
        for m, prefix in modalities.items():
            src_path = Path(base_dir) / f"{train_prefix}_{m}" / f"{prefix}{pid}.tif"
            dst_path = Path(base_dir) / f"{valid_prefix}_{m}" / f"{prefix}{pid}.tif"

            if src_path.exists():
                shutil.move(str(src_path), str(dst_path))
            else:
                print(f"⚠️ Warning: Missing {src_path}")

if __name__ == "__main__":
    dataset_dir = "/mnt/d/95-Cloud_training/"  # TODO: set your dataset root path
    split_train_valid(dataset_dir, split_ratio=0.8)
