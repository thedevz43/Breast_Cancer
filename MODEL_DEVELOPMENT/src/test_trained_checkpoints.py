import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path

files_benign = glob.glob(r"c:\Users\Admin\Desktop\Breast\RAW\BUS-UCLM\benign\*.png") + glob.glob(r"c:\Users\Admin\Desktop\Breast\RAW\BrEaST-Lesions_USG-images_and_masks-Dec-15-2023\**\*benign*.png", recursive=True)
files_malignant = glob.glob(r"c:\Users\Admin\Desktop\Breast\RAW\BUS-UCLM\malignant\*.png") + glob.glob(r"c:\Users\Admin\Desktop\Breast\RAW\BrEaST-Lesions_USG-images_and_masks-Dec-15-2023\**\*malignant*.png", recursive=True)

ckpt_path = r"c:\Users\Admin\Desktop\Breast\MODEL_DEVELOPMENT\checkpoints\m4_mobilenetv3_small_a_to_b_final.pt"

if Path(ckpt_path).exists():
    print(f"[+] Loading trained checkpoint {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    print("Keys in checkpoint:", list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt))
    
    # Inspect model architecture or state_dict
    state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt
    print(f"Loaded state_dict with {len(state_dict)} weight tensors.")
