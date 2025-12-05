import logging
import os
from typing import Optional, List

import cv2
import numpy as np
import torch
import yaml

from models import get_model


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)


class Segmentator:
    """Wrapper for loading a segmentation model and running inference."""

    def __init__(self, hparams: str, ckpt_path: str) -> None:
        self.cfg = self._load_config(hparams)
        self.model = self._load_model(self.cfg, ckpt_path)

    # ------------------------- CONFIG & MODEL LOADING -------------------------

    @staticmethod
    def _load_config(hparams: str) -> dict:
        """Load YAML configuration file."""
        with open(hparams, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def _load_model(self, cfg: dict, ckpt_path: str) -> torch.nn.Module:
        """Load model architecture and weights from checkpoint."""
        model_class = get_model(cfg["xp_config"]["model_name"])
        model = model_class(**cfg["model"])

        checkpoint = torch.load(ckpt_path, weights_only=True)

        # Detect the correct key for weights
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint:
                state_dict = checkpoint[key]
                break
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.eval()
        logging.info("Model loaded successfully.")
        return model

    # ------------------------- INFERENCE -------------------------

    def predict(self, batch: torch.Tensor, threshold: Optional[float] = None) -> np.ndarray:
        """
        Run inference on a batch.

        Args:
            batch: Input tensor of shape (B, C, H, W).
            threshold: Optional threshold for binarizing predictions.

        Returns:
            Numpy array of shape (B, H, W).
        """
        with torch.no_grad():
            logits = self.model(batch).squeeze(1)  # [B, H, W]
            probs = torch.sigmoid(logits).detach()

            if threshold is not None:
                probs = (probs >= threshold).to(probs.dtype)

        return probs.cpu().numpy()

    @staticmethod
    def save_prediction(pred: np.ndarray, save_path: str) -> None:
        """
        Save prediction mask as PNG.
        """
        pred = (pred.squeeze() * 255).astype(np.uint8)  # scale to [0,255]
        cv2.imwrite(save_path, pred)

    # ------------------------- DATA LOADING -------------------------

    def get_batch(self, root_path: str) -> torch.Tensor:
        """
        Load a batch from a folder containing subfolders like test_nir, test_red, ...
        Follows the channel order specified in cfg['data']['train_colors'] (ignores *_gt).

        Args:
            root_path: Path to folder containing subfolders.

        Returns:
            Tensor of shape (1, C, H, W).
        """
        colors = self.cfg["data"]["train_colors"]
        color_order = [c for c in colors if not c.endswith("gt")]
        ordered_files = []

        for color in color_order:
            color_suffix = color.replace("train_", "")
            subfolder = os.path.join(root_path, f"test_{color_suffix}")
            if not os.path.isdir(subfolder):
                raise FileNotFoundError(f"Missing subfolder: {subfolder}")

            files = [
                os.path.join(subfolder, f)
                for f in os.listdir(subfolder)
                if not os.path.isdir(os.path.join(subfolder, f))
            ]
            if not files:
                raise FileNotFoundError(f"No images found in {subfolder}")

            ordered_files.append(files[0])  # assume 1 image per channel

        ref_img = cv2.imread(ordered_files[0], cv2.IMREAD_GRAYSCALE)
        x = torch.empty((len(ordered_files), *ref_img.shape), dtype=torch.float32)

        for i, file in enumerate(ordered_files):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            x[i] = torch.from_numpy(img.astype(np.float32) / 255.0)

        return x.unsqueeze(0)

    def get_single_batch(self, root_path: str, image_id: str) -> torch.Tensor:
        """
        Load a single multi-channel batch given an image_id (e.g. "KS001_91294").
        It collects channels according to the training color order.

        Args:
            root_path: Path to folder containing test_* subfolders.
            image_id: Identifier of the image (without color prefix).

        Returns:
            Tensor of shape (1, C, H, W).
        """
        colors = self.cfg["data"]["train_colors"]
        color_order = [c for c in colors if not c.endswith("gt")]
        ordered_files = []

        for color in color_order:
            color_suffix = color.replace("train_", "")
            subfolder = os.path.join(root_path, f"test_{color_suffix}")

            matches = [
                f for f in os.listdir(subfolder)
                if f.startswith(f"{color_suffix}_{image_id}")
            ]
            if not matches:
                raise FileNotFoundError(f"No file found for {color_suffix}_{image_id} in {subfolder}")

            ordered_files.append(os.path.join(subfolder, matches[0]))

        ref_img = cv2.imread(ordered_files[0], cv2.IMREAD_GRAYSCALE)
        x = torch.empty((len(ordered_files), *ref_img.shape), dtype=torch.float32)

        for i, file in enumerate(ordered_files):
            img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
            x[i] = torch.from_numpy(img.astype(np.float32) / 255.0)

        return x.unsqueeze(0)

    def get_all_ids(self, root_path: str) -> List[str]:
        """
        Extract all unique image IDs from the dataset folder.
        Example filename: red_KS001_91294.tif -> ID = "KS001_91294".

        Args:
            root_path: Path to folder containing test_* subfolders.

        Returns:
            List of unique image IDs.
        """
        colors = self.cfg["data"]["train_colors"]
        first_color = [c for c in colors if not c.endswith("gt")][0]
        color_suffix = first_color.replace("train_", "")
        subfolder = os.path.join(root_path, f"test_{color_suffix}")

        if not os.path.isdir(subfolder):
            raise FileNotFoundError(f"Missing subfolder: {subfolder}")

        ids = []
        for f in os.listdir(subfolder):
            if not os.path.isdir(os.path.join(subfolder, f)):
                parts = f.split("_", 1)  # split once
                if len(parts) == 2:
                    ids.append(os.path.splitext(parts[1])[0])

        return sorted(set(ids))
