import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class MobileUNet(nn.Module):
    """Lightweight U-Net with MobileNet-style channel width for lesion segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = ConvBlock(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = ConvBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = ConvBlock(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)

class LightweightUNet(nn.Module):
    """Ultra-lightweight U-Net with narrow feature width for fast inference"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = ConvBlock(in_channels, 16)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(16, 32))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up1 = ConvBlock(128, 64)
        self.up2 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up2 = ConvBlock(64, 32)
        self.up3 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.conv_up3 = ConvBlock(32, 16)
        
        self.outc = nn.Conv2d(16, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x = torch.cat([x, x3], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)

def get_segmentation_model(name='MobileUNet'):
    if name == 'MobileUNet':
        return MobileUNet()
    elif name == 'LightweightUNet':
        return LightweightUNet()
    else:
        raise ValueError(f"Unknown model: {name}")
