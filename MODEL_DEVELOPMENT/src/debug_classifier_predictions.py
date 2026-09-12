import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from step17_m5_architecture import M5NextGenClassifier

cls_model = M5NextGenClassifier()
ckpt = torch.load(r"c:\Users\Admin\Desktop\Breast\MODEL_DEVELOPMENT\checkpoints\M6_FINAL\m6_mobilenetv3_small.pt", map_location='cpu', weights_only=False)
cls_model.load_state_dict(ckpt['state_dict'])
cls_model.eval()

files_spain = glob.glob(r"c:\Users\Admin\Desktop\Breast\RAW\BUS-UCLM\**\*.png", recursive=True) + glob.glob(r"c:\Users\Admin\Desktop\Breast\RAW\BUS-UCLM\**\*.jpg", recursive=True)
files_poland = glob.glob(r"c:\Users\Admin\Desktop\Breast\RAW\BrEaST-Lesions_USG-images_and_masks-Dec-15-2023\**\*.png", recursive=True)

test_files = files_spain[:5] + files_poland[:5]

for f in test_files:
    img = Image.open(f).convert('RGB').resize((224, 224))
    t = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        logits, qc = cls_model(t, t, t, t)
        prob = torch.sigmoid(logits).item()
        
    print(f"File: {Path(f).name} | Logits: {logits.item():.4f} | Raw Prob: {prob:.4f} | QC Score: {qc.item():.4f}")
