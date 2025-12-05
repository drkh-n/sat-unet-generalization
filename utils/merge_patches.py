import os
import re
import argparse
import numpy as np
import cv2

def merge_patches(input_dir, base_name, output_path, patch_size=384):
    # Match files like KS0017fbMI_006_000_patch_0_1.png
    pattern = re.compile(rf"{re.escape(base_name)}_patch_(\d+)_(\d+)\.png")
    
    patch_map = {}
    max_i = max_j = 0

    # Collect patches
    for fname in os.listdir(input_dir):
        match = pattern.match(fname)
        if match:
            i, j = int(match.group(1)), int(match.group(2))
            path = os.path.join(input_dir, fname)
            img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise ValueError(f"Failed to read patch: {path}")
            patch_map[(i, j)] = img
            max_i = max(max_i, i)
            max_j = max(max_j, j)

    if not patch_map:
        raise ValueError(f"No patches found for base name: {base_name}")

    patch_h, patch_w = patch_size, patch_size
    channels = list(patch_map.values())[0].shape[2] if len(patch_map.values().__iter__().__next__().shape) == 3 else 1

    # Create blank canvas
    height = (max_i + 1) * patch_h
    width = (max_j + 1) * patch_w
    merged = np.zeros((height, width, channels), dtype=np.uint8) if channels > 1 else np.zeros((height, width), dtype=np.uint8)

    # Paste patches
    for (i, j), img in patch_map.items():
        y, x = i * patch_h, j * patch_w
        if channels > 1:
            merged[y:y+patch_h, x:x+patch_w, :] = img
        else:
            merged[y:y+patch_h, x:x+patch_w] = img

    # Save merged output
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, merged)
    print(f"✅ Merged image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Folder with patch images')
    parser.add_argument('--base_name', type=str, required=True, help='Base name to match patch files (no _patch_*)')
    parser.add_argument('--output', type=str, required=True, help='Path to save the merged output image')
    args = parser.parse_args()

    merge_patches(args.input_dir, args.base_name, args.output)
