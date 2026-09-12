import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import torchvision.transforms as T

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
RAW_DIR = BASE_DIR / "RAW"

def get_crop_bbox(mask_np, margin_ratio=0.10):
    y_indices, x_indices = np.where(mask_np > 0)
    if len(x_indices) == 0:
        return None
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    
    margin_w = int(w * margin_ratio)
    margin_h = int(h * margin_ratio)
    
    img_h, img_w = mask_np.shape
    x1 = max(0, x_min - margin_w)
    y1 = max(0, y_min - margin_h)
    x2 = min(img_w, x_max + margin_w + 1)
    y2 = min(img_h, y_max + margin_h + 1)
    
    return (x1, y1, x2, y2)

class Step8AblationDataset(Dataset):
    def __init__(self, df, strategy='full_image', is_train=True, img_size=224):
        self.df = df.reset_index(drop=True)
        self.strategy = strategy.lower() # 'full_image', 'lesion_crop', 'full_plus_mask', 'full_plus_crop'
        self.is_train = is_train
        self.img_size = img_size
        
        self.norm_transform = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = RAW_DIR / str(row['relative_path'])
        mask_path = RAW_DIR / str(row['mask_path']) if pd.notna(row.get('mask_path')) and row.get('mask_path') else None
        
        # 1. Load RGB Image
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
        except Exception:
            img_rgb = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))

        # 2. Load Mask if available
        mask_np = None
        if mask_path and mask_path.exists():
            try:
                with Image.open(mask_path) as m_img:
                    mask_np = np.array(m_img.convert('L'))
            except Exception:
                mask_np = None

        w, h = img_rgb.size
        if mask_np is None:
            mask_np = np.zeros((h, w), dtype=np.uint8)

        # Binary label
        label_val = 1.0 if str(row.get('lesion_status', '')).lower() == 'malignant' or str(row.get('label_binary', '')) == '1' else 0.0
        label_tensor = torch.tensor(label_val, dtype=torch.float32)

        # Generate representations based on strategy
        if self.strategy == 'full_image':
            img_resized = img_rgb.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            t_img = T.ToTensor()(img_resized)
            t_img = self.norm_transform(t_img)
            return t_img, label_tensor

        elif self.strategy == 'lesion_crop':
            bbox = get_crop_bbox(mask_np)
            if bbox:
                x1, y1, x2, y2 = bbox
                crop_img = img_rgb.crop((x1, y1, x2, y2))
            else:
                crop_img = img_rgb
            crop_resized = crop_img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            t_crop = T.ToTensor()(crop_resized)
            t_crop = self.norm_transform(t_crop)
            return t_crop, label_tensor

        elif self.strategy == 'full_plus_mask':
            img_resized = img_rgb.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            mask_pil = Image.fromarray(mask_np).resize((self.img_size, self.img_size), Image.Resampling.NEAREST)
            
            t_img = T.ToTensor()(img_resized)
            t_img = self.norm_transform(t_img)
            t_mask = T.ToTensor()(mask_pil) # (1, H, W)
            
            # Concatenate 3-ch RGB + 1-ch Mask = 4-ch Tensor
            t_4ch = torch.cat([t_img, t_mask], dim=0)
            return t_4ch, label_tensor

        elif self.strategy == 'full_plus_crop':
            img_resized = img_rgb.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            t_full = T.ToTensor()(img_resized)
            t_full = self.norm_transform(t_full)
            
            bbox = get_crop_bbox(mask_np)
            if bbox:
                x1, y1, x2, y2 = bbox
                crop_img = img_rgb.crop((x1, y1, x2, y2))
            else:
                crop_img = img_rgb
            crop_resized = crop_img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
            t_crop = T.ToTensor()(crop_resized)
            t_crop = self.norm_transform(t_crop)
            
            return (t_full, t_crop), label_tensor

        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

import pandas as pd
