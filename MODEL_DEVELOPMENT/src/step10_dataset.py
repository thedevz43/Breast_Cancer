import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
import pandas as pd
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

class Step10ExternalDataset(Dataset):
    def __init__(self, df, img_size=224):
        self.df = df.reset_index(drop=True)
        self.img_size = img_size
        self.norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = RAW_DIR / str(row['relative_path'])
        mask_path = RAW_DIR / str(row['mask_path']) if pd.notna(row.get('mask_path')) and row.get('mask_path') else None
        
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
        except Exception:
            img_rgb = Image.new('RGB', (self.img_size, self.img_size), (0, 0, 0))
            
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
            
        label_val = 1.0 if str(row.get('lesion_status', '')).lower() == 'malignant' or str(row.get('label_binary', '')) == '1' else 0.0
        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        
        # 1. Full Image Branch
        img_full = img_rgb.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        t_full = self.norm(T.ToTensor()(img_full))
        
        # 2. Lesion Crop Branch
        bbox = get_crop_bbox(mask_np)
        if bbox:
            x1, y1, x2, y2 = bbox
            crop_img = img_rgb.crop((x1, y1, x2, y2))
        else:
            crop_img = img_rgb
        crop_resized = crop_img.resize((self.img_size, self.img_size), Image.Resampling.LANCZOS)
        t_crop = self.norm(T.ToTensor()(crop_resized))
        
        return t_full, t_crop, label_tensor, str(row.get('patient_id', '')), str(row.get('image_id', ''))
