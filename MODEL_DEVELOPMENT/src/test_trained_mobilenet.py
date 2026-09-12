import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from step8_models import build_full_mobilenet_v3_small

model = build_full_mobilenet_v3_small()
ckpt_path = r"c:\Users\Admin\Desktop\Breast\MODEL_DEVELOPMENT\checkpoints\m4_mobilenetv3_small_a_to_b_final.pt"

ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
state_dict = ckpt['state_dict'] if isinstance(ckpt, dict) and 'state_dict' in ckpt else ckpt

new_state_dict = {}
for k, v in state_dict.items():
    name = k.replace("module.", "")
    new_state_dict[name] = v

try:
    model.load_state_dict(new_state_dict)
    print("[OK] Successfully loaded trained weights into build_full_mobilenet_v3_small!")
except Exception as e:
    print("[-] Load failed:", e)

model.eval()

raw_dir = Path(r"c:\Users\Admin\Desktop\Breast\RAW")
all_files = [p for p in raw_dir.glob("**/*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]

benign_files = [p for p in all_files if "benign" in str(p).lower() or "ben" in str(p).lower()][:10]
malignant_files = [p for p in all_files if "malignant" in str(p).lower() or "mal" in str(p).lower() or "carcinoma" in str(p).lower()][:10]

print(f"\n[+] Found {len(benign_files)} Benign images and {len(malignant_files)} Malignant images.")

print("\n--- BENIGN SAMPLE PREDICTIONS ---")
for f in benign_files:
    img = Image.open(f).convert('RGB').resize((224, 224))
    t = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(t)
        prob = torch.sigmoid(out).item()
        
    # Platt Calibration (Slope=1.1842, Intercept=-0.0921)
    logit_val = np.log(prob / (1.0 - prob + 1e-7))
    prob_calib = float(np.clip(1.0 / (1.0 + np.exp(-(1.1842 * logit_val - 0.0921))), 0.001, 0.999))
    print(f"Benign {f.name}: Raw Prob = {prob*100:.1f}% | Calib Prob = {prob_calib*100:.1f}%")

print("\n--- MALIGNANT SAMPLE PREDICTIONS ---")
for f in malignant_files:
    img = Image.open(f).convert('RGB').resize((224, 224))
    t = torch.tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        out = model(t)
        prob = torch.sigmoid(out).item()
        
    logit_val = np.log(prob / (1.0 - prob + 1e-7))
    prob_calib = float(np.clip(1.0 / (1.0 + np.exp(-(1.1842 * logit_val - 0.0921))), 0.001, 0.999))
    print(f"Malignant {f.name}: Raw Prob = {prob*100:.1f}% | Calib Prob = {prob_calib*100:.1f}%")
