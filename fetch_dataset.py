import os
import glob
import pydicom
import numpy as np
import cv2
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset, DataLoader, random_split
import plistlib
import torch
import torch.nn as nn
import torch.nn.functional as F

class PreprocessedCalciumDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        # Count how many patient volume files exist in the folder
        self.num_samples = len([f for f in os.listdir(data_dir) if f.startswith("vol_")])
        
    def __len__(self):
        return self.num_samples
        
    def __getitem__(self, idx):
        # Load the pre-calculated tensors directly into memory
        vol_path = os.path.join(self.data_dir, f"vol_{idx}.pt")
        mask_path = os.path.join(self.data_dir, f"mask_{idx}.pt")
        
        # weights_only=True is safer and slightly faster in modern PyTorch
        vol = torch.load(vol_path, weights_only=True)
        mask = torch.load(mask_path, weights_only=True)

        # 1. ENSURE PYTORCH TENSORS
        if not isinstance(vol, torch.Tensor):
            vol = torch.tensor(vol, dtype=torch.float32)
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask, dtype=torch.float32)
    
        # 2. FIX THE DIMENSIONS (Crucial for MONAI)
        if vol.dim() == 3:
            vol = vol.unsqueeze(0)
        if mask.dim() == 3:
            mask = mask.unsqueeze(0)
        
        # 3. Package them into a dictionary for MONAI
        data_dict = {"image": vol, "label": mask}

        # 4. Apply the transform if it exists
        if self.transform is not None:
            data_dict = self.transform(data_dict)

        # 5. UNPACK AND HANDLE MULTIPLE PATCHES
        # If num_samples > 1, MONAI returns a list of dictionaries.
        if isinstance(data_dict, list):
            # Return a list of tuples: [(img1, mask1), (img2, mask2), ...]
            return [(d["image"], d["label"]) for d in data_dict]
        else:
            # Transform returned a single patch (or no transform applied)
            return data_dict["image"], data_dict["label"]


# --- NEW ADDITION: The Custom Collator ---
def calcium_collate_fn(batch):
    """
    This intercepts the batch from the Dataset before it hits the training loop.
    It flattens the lists so that if a batch size of 2 patients yields 4 patches each,
    it outputs a clean, unified batch of 8 patches for the GPU.
    """
    images = []
    labels = []
    
    for item in batch:
        if isinstance(item, list):
            # If the item is a list of patches from one patient
            for img, lbl in item:
                images.append(img)
                labels.append(lbl)
        else:
            # If the item is just a single patch
            img, lbl = item
            images.append(img)
            labels.append(lbl)
            
    # Stack them into standard 5D tensors: (Batch, Channel, Depth, Height, Width)
    return torch.stack(images), torch.stack(labels)