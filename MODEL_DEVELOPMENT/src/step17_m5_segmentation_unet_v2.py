import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    """Attention Gate to focus on salient lesion boundary regions"""
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

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

class MobileUNetV2(nn.Module):
    """Mobile U-Net v2 with Attention Gates for High-Precision Lesion Segmentation"""
    def __init__(self, in_channels=3, out_channels=1):
        super().__init__()
        self.inc = ConvBlock(in_channels, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(128, 256))
        
        self.ag1 = AttentionGate(F_g=128, F_l=128, F_int=64)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = ConvBlock(256, 128)
        
        self.ag2 = AttentionGate(F_g=64, F_l=64, F_int=32)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = ConvBlock(128, 64)
        
        self.ag3 = AttentionGate(F_g=32, F_l=32, F_int=16)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = ConvBlock(64, 32)
        
        self.outc = nn.Conv2d(32, out_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        x = self.up1(x4)
        x3_att = self.ag1(g=x, x=x3)
        x = torch.cat([x, x3_att], dim=1)
        x = self.conv_up1(x)
        
        x = self.up2(x)
        x2_att = self.ag2(g=x, x=x2)
        x = torch.cat([x, x2_att], dim=1)
        x = self.conv_up2(x)
        
        x = self.up3(x)
        x1_att = self.ag3(g=x, x=x1)
        x = torch.cat([x, x1_att], dim=1)
        x = self.conv_up3(x)
        
        logits = self.outc(x)
        return torch.sigmoid(logits)

class CompoundBoundaryLoss(nn.Module):
    def __init__(self, lambda1=1.0, lambda2=1.0, lambda3=0.5):
        super().__init__()
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.lambda3 = lambda3
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        bce_loss = self.bce(pred, target)
        smooth = 1e-5
        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice_loss = 1.0 - ((2.0 * intersection + smooth) / (union + smooth)).mean()
        
        laplacian_kernel = torch.tensor([[[[-1., -1., -1.], [-1., 8., -1.], [-1., -1., -1.]]]], device=pred.device)
        pred_edge = F.conv2d(pred, laplacian_kernel, padding=1)
        target_edge = F.conv2d(target, laplacian_kernel, padding=1)
        boundary_loss = F.l1_loss(pred_edge, target_edge)
        
        total_loss = (self.lambda1 * dice_loss) + (self.lambda2 * bce_loss) + (self.lambda3 * boundary_loss)
        return total_loss
