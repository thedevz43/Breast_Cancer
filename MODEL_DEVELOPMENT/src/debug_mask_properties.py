import glob
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from step17_m5_segmentation_unet_v2 import MobileUNetV2

seg_model = MobileUNetV2()
ckpt = torch.load(r"c:\Users\Admin\Desktop\Breast\MODEL_DEVELOPMENT\checkpoints\M6_FINAL\mobile_unet_v2_segmentation.pt", map_location='cpu', weights_only=False)
seg_model.load_state_dict(ckpt['state_dict'])
seg_model.eval()

files = glob.glob(r"c:\Users\Admin\Desktop\Breast\RAW\**\*.png", recursive=True)[:10]

for f in files:
    img = Image.open(f).convert('RGB').resize((224, 224))
    tensor = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = seg_model(tensor)
        prob = torch.sigmoid(out)
        mask = (prob.squeeze().numpy() > 0.5).astype(np.uint8)
        
    area_ratio = mask.sum() / mask.size
    print(f"File: {Path(f).name} | Mask Pixels > 0.5: {mask.sum()} ({area_ratio*100:.2f}%) | Prob Max: {prob.max():.4f} | Prob Mean: {prob.mean():.4f}")
