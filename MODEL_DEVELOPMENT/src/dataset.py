import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
RAW_DIR = BASE_DIR / "RAW"

class BreastUltrasoundDataset(Dataset):
    def __init__(self, df, transform=None, is_classical=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_classical = is_classical
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = RAW_DIR / row['relative_path']
        
        try:
            with Image.open(img_path) as img:
                img_rgb = img.convert('RGB')
        except Exception:
            # Fallback black image if missing/corrupt
            img_rgb = Image.new('RGB', (224, 224), (0, 0, 0))
            
        if self.transform:
            img_tensor = self.transform(img_rgb)
        else:
            img_tensor = img_rgb
            
        # Binary label (0: Benign, 1: Malignant)
        label_val = float(row['label_binary']) if str(row['label_binary']).isdigit() or str(row['label_binary']).replace('.','',1).isdigit() else (1.0 if str(row['lesion_status']).lower()=='malignant' else 0.0)
        label_tensor = torch.tensor(label_val, dtype=torch.float32)
        
        if self.is_classical:
            # Flatten image tensor to 1D vector (3 * 224 * 224 = 150,528 features)
            flat_feat = img_tensor.view(-1).numpy()
            return flat_feat, int(label_val)
            
        return img_tensor, label_tensor
