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
    def __init__(self, data_dir):
        self.data_dir = data_dir
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
        
        return vol, mask
'''
class CoronaryCalciumDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir: Path to the 'Gated_release_final' directory.
        """
        self.data_dir = data_dir
        self.patient_dir = os.path.join(data_dir, "patient")
        self.xml_dir = os.path.join(data_dir, "calcium_xml")
        
        # Get all patient IDs (the folders '0', '1', '2', etc.)
        # We ensure we are only grabbing directories, ignoring hidden files like .DS_Store
        self.patient_ids = [d for d in os.listdir(self.patient_dir) 
                            if os.path.isdir(os.path.join(self.patient_dir, d))]
        
        self.transform = transform
        self.hu_min = -100.0
        self.hu_max = 800.0

    def __len__(self):
        return len(self.patient_ids)

    def load_dicom_volume(self, dicom_dir):
        # Read all files in the directory since extensions vary (e.g., .180)
        file_list = os.listdir(dicom_dir)
        slices = []
        
        for f in file_list:
            file_path = os.path.join(dicom_dir, f)
            try:
                # Try to read as DICOM. Will skip hidden/system files automatically.
                ds = pydicom.dcmread(file_path)
                # Only keep files that actually have image data
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)
            except pydicom.errors.InvalidDicomError:
                continue
                
        if len(slices) == 0:
            raise ValueError(f"No valid DICOM images found in {dicom_dir}")
        
        # Sort slices by Z-position (head-to-toe)
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        
        # Convert to Hounsfield Units (HU)
        image_volume = np.stack([s.pixel_array for s in slices])
        image_volume = image_volume.astype(np.float32)
        
        intercept = getattr(slices[0], 'RescaleIntercept', 0)
        slope = getattr(slices[0], 'RescaleSlope', 1)
        image_volume = image_volume * slope + intercept
        
        return image_volume, slices

    def load_xml_segmentation(self, xml_path, depth, height, width):
        mask_volume = np.zeros((depth, height, width), dtype=np.uint8)
        
        if not os.path.exists(xml_path):
            return mask_volume 
            
        # Parse the Apple .plist XML file into a Python Dictionary
        with open(xml_path, 'rb') as f:
            try:
                plist_data = plistlib.load(f)
            except Exception as e:
                print(f"Failed to read plist file {xml_path}: {e}")
                return mask_volume
                
        # Get the list of images (slices) that have ROIs
        images = plist_data.get('Images', [])
        
        for image_data in images:
            z_index = image_data.get('ImageIndex', -1)
            
            # Prevent out-of-bounds indexing
            if z_index < 0 or z_index >= depth:
                continue
                
            rois = image_data.get('ROIs', [])
            for roi in rois:
                # We extract the 'Point_px' array from the plist
                # It looks like: ['(269.003, 377.000)', '(269.000, 377.003)', ...]
                point_strings = roi.get('Point_px', [])
                points = []
                
                for pt_str in point_strings:
                    # Clean the string by removing parentheses and splitting by comma
                    clean_str = pt_str.replace('(', '').replace(')', '')
                    x_str, y_str = clean_str.split(',')
                    points.append([float(x_str), float(y_str)])
                
                # If we successfully parsed a polygon, draw it!
                if len(points) > 2:
                    points = np.array(points, dtype=np.int32)
                    cv2.fillPoly(mask_volume[z_index], [points], color=1)
                    
        return mask_volume

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        
        # --- 1. Locate the correct DICOM folder ---
        patient_base_path = os.path.join(self.patient_dir, patient_id)
        
        # Find the randomly named subfolder inside the patient folder
        subfolders = [f for f in os.listdir(patient_base_path) 
                      if os.path.isdir(os.path.join(patient_base_path, f))]
        
        if not subfolders:
            raise FileNotFoundError(f"Patient folder {patient_id} is empty!")
            
        dicom_dir = os.path.join(patient_base_path, subfolders[0])
        
        # --- 2. Locate the corresponding XML file ---
        # Assuming the XML is named exactly like the patient folder (e.g., "0.xml")
        xml_path = os.path.join(self.xml_dir, f"{patient_id}.xml")
        
        # --- 3. Process Data ---
        volume, dicom_slices = self.load_dicom_volume(dicom_dir)
        depth, height, width = volume.shape
        
        # Normalize HU values to [0, 1]
        volume = np.clip(volume, self.hu_min, self.hu_max)
        volume = (volume - self.hu_min) / (self.hu_max - self.hu_min)
        
        # Load mask
        mask = self.load_xml_segmentation(xml_path, depth, height, width)
        
        # Convert to PyTorch Tensors
        volume_tensor = torch.tensor(volume, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.long)
        
        # Interpolate to fixed size for batching (e.g., 64 slices, 128x128 resolution)
        fixed_depth = 64
        fixed_hw = 128
        
        volume_tensor = volume_tensor.unsqueeze(0).unsqueeze(0)
        mask_tensor = mask_tensor.unsqueeze(0).unsqueeze(0).float()
        
        volume_tensor = F.interpolate(volume_tensor, size=(fixed_depth, fixed_hw, fixed_hw), mode='trilinear', align_corners=False)
        mask_tensor = F.interpolate(mask_tensor, size=(fixed_depth, fixed_hw, fixed_hw), mode='nearest')
        
        volume_tensor = volume_tensor.squeeze(0).squeeze(0)
        mask_tensor = mask_tensor.squeeze(0).squeeze(0).long()
        
        return volume_tensor, mask_tensor
'''


