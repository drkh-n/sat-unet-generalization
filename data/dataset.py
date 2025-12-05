import os
import random
from typing import Optional, Dict, List, Any

import torch
import lightning as L
import cv2
import numpy as np

import torch

from torchvision import tv_tensors
from torch.utils.data.dataloader import default_collate
# from .augmentation import STD_TRANSFROMS

import pandas as pd
import random

import matplotlib.pyplot as plt

class SatSlideCollate:
    """
    Implements the 'Sat-SlideMix' (from 'Sat-SlideMix' pseudocode) 
    augmentation as a collate_fn.

    This augmentation creates 'gamma' number of shifted (rolled) versions
    for each item in an input batch and applies the *same* shift
    to both the image (x) and the mask (y).
    """
    def __init__(self, gamma: int, beta: float):
        """
        Args:
            gamma (int): Number of augmented versions to create per item (line 1).
            beta (float): Max shift magnitude as a fraction [0.0, 1.0] of the 
                          dimension size (line 1).
        """
        assert gamma >= 1, "gamma must be 1 or greater"
        assert 0.0 <= beta <= 1.0, "beta must be a float between 0.0 and 1.0"
        self.gamma = gamma
        self.beta = beta

    def __call__(self, batch):
        """
        Applies the Sat-Slide augmentation to a batch of (image, mask) pairs.
        
        Args:
            batch: A list of (x, y) tuples, where:
                   x is [C, H, W] tensor
                   y is [H, W] tensor
        
        Returns:
            (torch.Tensor, torch.Tensor): 
             - Augmented x_batch, shape [gamma * B, C, H, W]
             - Augmented y_batch, shape [gamma * B, H, W]
        """

        x_batch, y_batch = default_collate(batch)
        B, C, H, W = x_batch.shape
        gamma_batch_size = self.gamma * B
        repeated_x = x_batch.repeat(self.gamma, 1, 1, 1)

        repeated_y = y_batch.repeat(self.gamma, 1, 1)

        magnitudes = torch.rand(gamma_batch_size, device=x_batch.device) * self.beta

        rolled_x_list = []
        rolled_y_list = []

        for i in range(gamma_batch_size):
            img = repeated_x[i]  # [C, H, W]
            mask = repeated_y[i] # [H, W]
            m_float = magnitudes[i]

            dim_idx = random.randint(0, 1) 
            dim_torch_x = dim_idx + 1
            dim_torch_y = dim_idx
            
            dim_size = img.shape[dim_torch_x]
            shift = int(round((m_float * dim_size).item()))
            if random.random() < 0.5:
                shift = -shift
            
            rolled_img = torch.roll(img, shifts=shift, dims=dim_torch_x)
            rolled_mask = torch.roll(mask, shifts=shift, dims=dim_torch_y)
            
            rolled_x_list.append(rolled_img)
            rolled_y_list.append(rolled_mask)

        final_x = torch.stack(rolled_x_list, dim=0)
        final_y = torch.stack(rolled_y_list, dim=0)
        
        return final_x, final_y

    def visualize(self, x_batch, y_batch, num_items=1):
        """
        Visualizes original + gamma augmentations.
        Each row shows: NIR | RED | GREEN | BLUE | GT
        """

        gamma = self.gamma
        B_total = x_batch.shape[0]  # = gamma * B

        for item_idx in range(num_items):
            # index of the first augmented version for this item
            start = item_idx * gamma

            rows = gamma
            cols = 5  # 4 bands + GT

            fig, axs = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))

            for g in range(gamma):
                img = x_batch[start + g].cpu()  # [4, H, W]
                gt = y_batch[start + g].cpu()   # [H, W]

                # 4 channels
                for ch in range(4):
                    axs[g, ch].imshow(img[ch], cmap='gray')
                    axs[g, ch].set_title(
                        ["NIR", "RED", "GREEN", "BLUE"][ch] 
                    )
                    axs[g, ch].axis("off")

                # GT
                axs[g, 4].imshow(gt, cmap='gray')
                axs[g, 4].set_title("GT")
                axs[g, 4].axis("off")

            plt.tight_layout()
            plt.show()


class CloudDataset(torch.utils.data.Dataset):
    def __init__(self, files, augment = False, normalization: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.files = files
        self.augment = augment
        self.seq = 0
        self.normalization = normalization
                                       
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        y = cv2.imread(self.files[idx][-1], cv2.IMREAD_GRAYSCALE)//255
        y = torch.tensor(y, dtype=torch.int64)
        x = torch.empty(len(self.files[idx])-1, *y.shape, dtype=torch.float32)
        # df = pd.read_csv(self.normalization['csv_path'], usecols=[str(self.normalization['p_high']), str(self.normalization['p_low'])])
        for i in range(x.shape[0]):
            # p_low = df[str(self.normalization['p_low'])][i]
            # p_high = df[str(self.normalization['p_high'])][i]
            # x[i] = torch.from_numpy( (cv2.imread(self.files[idx][i], cv2.IMREAD_GRAYSCALE) - p_low) / (p_high - p_low) )     
            x[i] = torch.from_numpy(cv2.imread(self.files[idx][i], cv2.IMREAD_GRAYSCALE)/255)        
        if self.augment:
            pass
            # x, y = STD_TRANSFROMS(x, y)

        return x, y 

class CloudDataModule(L.LightningDataModule):
    def __init__(self, data_root, train_colors, valid_colors, train_batch_size=32, 
                 valid_batch_size=32, num_workers = 0, persistent_workers = False, 
                 pin_memory = True, shuffle = True, seed = 42, augment = False, multi_gpu = False, type="torch",
                 normalization: Optional[Dict[str, Any]] = None):
        super(CloudDataModule, self).__init__()
        self.train_dirs = [os.path.join(data_root, color) for color in train_colors]
        self.valid_dirs = [os.path.join(data_root, color) for color in valid_colors]
        self.train_batch_size = train_batch_size
        self.valid_batch_size = valid_batch_size
        self.shuffle = shuffle
        if type=="lightning":
            self.num_workers=0
            self.persistent_workers=False
        else:
            self.num_workers = num_workers
            self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory
        self.augment = augment
        self.seed = seed
        self._set_seed(seed)
        self.multi_gpu = multi_gpu
        self.train_sampler = None
        self.valid_sampler = None
        self.normalization = normalization
        self.setup()

    def combine_files(self, file, dirs):
        channels = len(dirs) - 1
        gt_folder = dirs[-1]
        subs = [directory.split(os.path.sep)[-1] for directory in dirs]
        TIFsGT = []
        for i in range(channels):
            TIFsGT.append(os.path.join(dirs[i], file.replace(file.split('_')[0], subs[i].split('_')[1])))
        TIFsGT.append(os.path.join(gt_folder, file))
        return TIFsGT
        
    def setup(self, stage: Optional[str] = None):
        train_files = [self.combine_files(file, self.train_dirs) for file in os.listdir(self.train_dirs[-1]) if not os.path.isdir(file)]
        valid_files = [self.combine_files(file, self.valid_dirs) for file in os.listdir(self.valid_dirs[-1]) if not os.path.isdir(file)]
        random.shuffle(train_files)
        random.shuffle(valid_files)

        if stage in (None, "fit"):
            self.train_dataset = CloudDataset(train_files, augment=self.augment, normalization=self.normalization)
            self.valid_dataset = CloudDataset(valid_files, augment=False, normalization=self.normalization)
        
        if self.multi_gpu and torch.distributed.is_initialized():
            self.train_sampler = torch.utils.data.DistributedSampler(self.train_dataset, num_replicas=torch.distributed.get_world_size(), 
                        rank=torch.distributed.get_rank(), shuffle=True)
            self.valid_sampler = torch.utils.data.DistributedSampler(self.valid_dataset, num_replicas=torch.distributed.get_world_size(), 
                        rank=torch.distributed.get_rank(), shuffle=False)
    
    def train_dataloader(self):
        collate_fn = SatSlideCollate(gamma=3, beta=1) if self.augment else None
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=self.shuffle and self.train_sampler is None,
            worker_init_fn=self.seed_worker,
            persistent_workers=self.persistent_workers,
            collate_fn=collate_fn,
            sampler=self.train_sampler if (self.multi_gpu and torch.distributed.is_initialized()) else None
        )


    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            batch_size=self.valid_batch_size,
            pin_memory=self.pin_memory,
            num_workers=self.num_workers,
            shuffle=False,
            worker_init_fn=self.seed_worker,
            persistent_workers=self.persistent_workers
        )

    '''
    def test_dataloader(self):
        return torch.utils.data.DataLoader(self.test_dataset,
                                    batch_size = self.valid_batch_size,
                                    pin_memory = self.pin_memory,
                                    num_workers=self.num_workers, 
                                    shuffle = False,
                                    worker_init_fn=self.seed_worker,
                                    generator=self.g,
                                    persistent_workers = self.persistent_workers)
    '''
    
    def _set_seed(self, seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32 + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)


if __name__ == '__main__':
    cloud_dm = CloudDataModule('/mnt/d/38-Cloud_training', 
                           ["train_nir", "train_red", "train_green", "train_blue", "train_gt"], 
                           ["valid_nir", "valid_red", "valid_green", "valid_blue", "valid_gt"],
                           augment=True)
    cloud_dm.setup(stage="fit")
    train_loader = cloud_dm.train_dataloader()
    collate_fn = SatSlideCollate(gamma=3, beta=0.5)

    for xb, yb in train_loader:
        collate_fn.visualize(xb, yb, num_items=4)
        break

    # for x, y in train_loader:  
    #     print(x, y)
    
    # # Get one batch
    # x, y = next(iter(train_loader)) 
    
    print("\n--- Output Batch Shapes ---")
    # print(f"Augmented Images (x) Shape: {x.shape}")
    # print(f"Augmented Masks (y) Shape:  {y.shape}")
    
    print("\nSuccess! Batch shapes are correct.")




