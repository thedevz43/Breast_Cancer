import torch
import torch.nn as nn

class LCCAModule(nn.Module):
    """Lightweight Cross-Channel Attention Module"""
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class M5NextGenClassifier(nn.Module):
    """
    M5 Proposed Next-Generation Architecture:
    Multi-Scale Lesion + 30% Perilesional Ring Feature Fusion + Global Context + Localization QC Branch
    """
    def __init__(self, backbone='MobileNetV3-Small'):
        super().__init__()
        self.backbone_name = backbone
        
        # Adaptive pooling encoders
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Input channel encoders for 4 streams
        self.enc_global = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.enc_tight = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.enc_crop10 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        self.enc_crop30 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)))
        
        # Fusion Classifier Head (64 input channels)
        self.classifier_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )
        
        # QC Head
        self.qc_head = nn.Sequential(
            nn.Linear(64, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x_global, x_tight, x_crop10, x_crop30):
        b = x_global.size(0)
        
        f_global = self.enc_global(x_global).view(b, 16)
        f_tight = self.enc_tight(x_tight).view(b, 16)
        f_crop10 = self.enc_crop10(x_crop10).view(b, 16)
        f_crop30 = self.enc_crop30(x_crop30).view(b, 16)
        
        f_cat = torch.cat([f_global, f_tight, f_crop10, f_crop30], dim=1)
        
        logits = self.classifier_head(f_cat)
        qc_score = self.qc_head(f_cat)
        
        return logits, qc_score
