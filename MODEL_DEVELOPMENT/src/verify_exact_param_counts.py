import torch
import torch.nn as nn
import pandas as pd
from pathlib import Path

from step9_models import get_step9_model
from step16_segmentation_models import MobileUNet
from cnn_baselines import get_cnn_model

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_all_params(model):
    return sum(p.numel() for p in model.parameters())

def audit_exact_parameter_counts():
    print("[+] Calculating EXACT PyTorch Parameter Counts for All Architectures...")
    
    models_to_test = [
        ("MobileNetV3-Small Baseline (Single Branch)", get_cnn_model('MobileNetV3-Small')),
        ("GhostNet Baseline (Single Branch)", get_cnn_model('GhostNet')),
        ("ResNet-18 Baseline (Single Branch)", get_cnn_model('ResNet-18')),
        ("Mobile U-Net (Segmentation Network)", MobileUNet()),
        ("M4-MobileNetV3-Small (Proposed Dual Branch)", get_step9_model('M4', 'MobileNetV3-Small')),
        ("M4-GhostNet (Dual Branch)", get_step9_model('M4', 'GhostNet')),
        ("M4-ResNet-18 (Dual Branch)", get_step9_model('M4', 'ResNet-18'))
    ]
    
    audit_rows = []
    for name, model in models_to_test:
        trainable_p = count_params(model)
        total_p = count_all_params(model)
        size_mb = (total_p * 4) / (1024 * 1024)
        
        audit_rows.append({
            'Architecture Name': name,
            'Trainable Parameter Count': trainable_p,
            'Total Parameter Count': total_p,
            'Formatted Count': f"{total_p / 1e6:.2f} M ({total_p:,} params)",
            'Model Size (MB)': round(size_mb, 2)
        })
        
    df_audit = pd.DataFrame(audit_rows)
    print(df_audit.to_string())
    
    csv_out = Path(r"c:\Users\Admin\Desktop\Breast\MODEL_DEVELOPMENT\exact_parameter_counts_audit.csv")
    df_audit.to_csv(csv_out, index=False)
    print(f"\n[OK] Exact parameter count audit saved to {csv_out}")

if __name__ == "__main__":
    audit_exact_parameter_counts()
