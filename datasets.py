"""PyTorch Dataset classes for the CT volumes and multi-channel pathology
tiles, plus their augmentation transforms."""
import os
import random
import time
from typing import Tuple, Dict

import numpy as np
from PIL import Image
import pydicom
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

from utils import extract_patient_id_from_name

class CT3DVolumeDataset(Dataset):
    def __init__(self, data_dir: str, depth: int = 16, slice_size: int = 128, max_volumes: int = None, transform=None, seed: int = 42):
        self.data_dir = data_dir
        self.depth = depth
        self.slice_size = slice_size
        self.transform = transform
        self.rng = np.random.RandomState(seed)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"CT data directory not found: {data_dir}")
        
        self.volumes = []
        self.labels = []
        self.patient_ids = []
        
        self._load_volumes()
        
        if len(self.volumes) == 0:
            raise ValueError(f"No valid CT volumes found in {data_dir}")
    
    def _load_volumes(self) -> None:
        log_message(f"Loading ALL CT volumes from {self.data_dir}")
        
        patient_dirs = sorted([
            d for d in os.listdir(self.data_dir)
            if os.path.isdir(os.path.join(self.data_dir, d)) and d.startswith('R_')
        ])
        
        if not patient_dirs:
            raise FileNotFoundError(f"No patient directories found in {self.data_dir}")
        
        total_volumes_loaded = 0
        
        for patient_id in tqdm(patient_dirs, desc="Loading CT volumes"):
            patient_path = os.path.join(self.data_dir, patient_id)
            patient_volume_count = 0
            
            for date_dir in os.listdir(patient_path):
                date_path = os.path.join(patient_path, date_dir)
                if not os.path.isdir(date_path):
                    continue
                
                for series_dir in os.listdir(date_path):
                    series_path = os.path.join(date_path, series_dir)
                    if not os.path.isdir(series_path):
                        continue
                    
                    dcm_files = sorted([
                        f for f in os.listdir(series_path)
                        if f.endswith('.dcm')
                    ])
                    
                    if len(dcm_files) < self.depth:
                        continue
                    
                    volume_slices = []
                    valid_count = 0
                    
                    for dcm_file in dcm_files:
                        if len(volume_slices) >= self.depth:
                            break
                            
                        dcm_path = os.path.join(series_path, dcm_file)
                        try:
                            ds = pydicom.dcmread(dcm_path)
                            pixel_array = ds.pixel_array.astype(np.float32)
                            pixel_array = np.clip(pixel_array, -1000, 1000)
                            pixel_array = (pixel_array + 1000) / 2000
                            slice_img = Image.fromarray(pixel_array)
                            slice_img = slice_img.resize((self.slice_size, self.slice_size), Image.BILINEAR)
                            slice_array = np.array(slice_img, dtype=np.float32)
                            volume_slices.append(slice_array)
                            valid_count += 1
                            
                        except Exception as e:
                            continue
                    
                    if valid_count >= self.depth:
                        volume = np.stack(volume_slices[:self.depth], axis=0)
                        self.volumes.append(volume)
                        # CT labels are intentionally unavailable in Stage-1 SSL.
                        label = -1
                        self.labels.append(label)
                        self.patient_ids.append(f"{patient_id}_series_{patient_volume_count}")
                        total_volumes_loaded += 1
                        patient_volume_count += 1
        
        log_message(f"Loaded {len(self.volumes)} CT volumes from {len(patient_dirs)} patients")
        log_message(f"Average volumes per patient: {len(self.volumes)/len(patient_dirs):.2f}")
    
    def __len__(self) -> int:
        return len(self.volumes)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        volume = self.volumes[idx].copy()
        label = self.labels[idx]
        patient_id = self.patient_ids[idx]
        
        if self.transform:
            volume = self.transform(volume, self.rng)
        
        volume_tensor = torch.from_numpy(volume).unsqueeze(0).float()
        
        return volume_tensor, label, patient_id

class PathologyMultiChannelDataset(Dataset):
    def __init__(self, data_dir: str, channels: Tuple[str, ...], image_size: int = 224, max_samples: int = None, transform=None, seed: int = 42):
        self.data_dir = data_dir
        self.channels = channels
        self.image_size = image_size
        self.transform = transform
        self.rng = np.random.RandomState(seed)
        
        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Pathology data directory not found: {data_dir}")
        
        self.samples = []
        self._load_samples()
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid pathology samples found in {data_dir}")
    
    def _load_samples(self) -> None:
        log_message(f"Loading ALL pathology samples from {self.data_dir}")
        
        seg_dir = os.path.join(self.data_dir, 'Segmentation')
        if not os.path.exists(seg_dir):
            raise FileNotFoundError(f"Segmentation directory not found: {seg_dir}")
        
        mask_files = sorted([f for f in os.listdir(seg_dir) if f.endswith('.png')])
        log_message(f"Found {len(mask_files)} total mask files")
        
        loaded_count = 0
        skipped_count = 0
        
        for mask_file in tqdm(mask_files, desc="Loading pathology samples"):
            base_name = mask_file.replace('.png', '')
            mask_path = os.path.join(seg_dir, mask_file)
            
            try:
                mask = Image.open(mask_path).convert('L')
                mask_array = np.array(mask)
                # Segmentation mask is used only to derive weak supervision labels.
                # ROI extraction is handled separately in a label-blind manner at __getitem__ time.
                tumor_ratio = (mask_array > 127).sum() / (mask_array.size + 1e-8)
                label = 1 if tumor_ratio > 0.050 else 0
                
                channel_paths = self._get_channel_paths(base_name)
                if len(channel_paths) == len(self.channels):
                    patient_id = extract_patient_id_from_name(base_name)
                    self.samples.append({
                        'base_name': base_name,
                        'patient_id': patient_id,
                        'label': label,
                        'mask_path': mask_path,
                        'channel_paths': channel_paths
                    })
                    loaded_count += 1
                else:
                    skipped_count += 1
                
            except Exception as e:
                skipped_count += 1
                continue
        
        log_message(f"Successfully loaded {len(self.samples)} pathology samples")
        log_message(f"Skipped {skipped_count} samples due to missing channels or errors")
    
    def _get_channel_paths(self, base_name: str) -> Dict[str, str]:
        channel_paths = {}
        
        for channel in self.channels:
            found = False
            for subdir in ['mIF_Data', 'mIHC_Data']:
                fname = f"{base_name}_{channel}.png"
                fpath = os.path.join(self.data_dir, subdir, fname)
                
                if os.path.isfile(fpath):
                    channel_paths[channel] = fpath
                    found = True
                    break
            
            if not found:
                return {}
        
        return channel_paths
    
    def _load_multichannel_image(self, channel_paths: Dict[str, str]) -> np.ndarray:
        channels_data = []
        
        for channel in self.channels:
            img = Image.open(channel_paths[channel]).convert('L')
            img = img.resize((self.image_size, self.image_size), Image.BILINEAR)
            img_array = np.array(img, dtype=np.float32) / 255.0
            channels_data.append(img_array)
        
        return np.stack(channels_data, axis=0)

    def _extract_label_blind_roi(self, image: np.ndarray) -> np.ndarray:
        # ROI is derived from signal intensity (DAPI preferred), not from diagnostic labels/masks.
        dapi_index = self.channels.index('DAPI') if 'DAPI' in self.channels else 0
        reference = image[dapi_index]
        threshold = np.percentile(reference, 75.0)
        roi_mask = (reference > threshold).astype(np.float32)

        if roi_mask.mean() < 0.01:
            fallback_threshold = reference.mean() + 0.3 * reference.std()
            roi_mask = (reference > fallback_threshold).astype(np.float32)

        return image * roi_mask[None, :, :]
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        sample = self.samples[idx]
        image = self._load_multichannel_image(sample['channel_paths'])
        image = self._extract_label_blind_roi(image)
        
        if self.transform:
            image = self.transform(image, self.rng)
        
        image_tensor = torch.from_numpy(image).float()
        label = sample['label']
        base_name = sample['base_name']
        
        return image_tensor, label, base_name

class CT3DAugmentation:
    def __init__(self, strength: float = 0.5):
        self.strength = strength
    
    def __call__(self, volume: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        if rng.random() < 0.5:
            volume = volume[::-1].copy()
        
        if rng.random() < 0.5:
            volume = volume[:, ::-1, :].copy()
        
        if rng.random() < 0.5:
            volume = volume[:, :, ::-1].copy()
        
        if rng.random() < self.strength:
            scale = rng.uniform(0.9, 1.1)
            volume = np.clip(volume * scale, 0, 1)
        
        if rng.random() < self.strength:
            shift = rng.uniform(-0.1, 0.1)
            volume = np.clip(volume + shift, 0, 1)
        
        if rng.random() < self.strength * 0.5:
            noise = rng.normal(0, 0.02, volume.shape)
            volume = np.clip(volume + noise, 0, 1)
        
        return volume

class Pathology2DAugmentation:
    def __init__(self, strength: float = 0.5):
        self.strength = strength
    
    def __call__(self, image: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
        if rng.random() < 0.5:
            image = image[:, :, ::-1].copy()
        
        if rng.random() < 0.5:
            image = image[:, ::-1, :].copy()
        
        if rng.random() < 0.5:
            k = rng.randint(1, 4)
            image = np.rot90(image, k, axes=(1, 2)).copy()
        
        if rng.random() < self.strength:
            for c in range(image.shape[0]):
                scale = rng.uniform(0.9, 1.1)
                image[c] = np.clip(image[c] * scale, 0, 1)
        
        if rng.random() < self.strength * 0.5:
            noise = rng.normal(0, 0.02, image.shape)
            image = np.clip(image + noise, 0, 1)
        
        return image
