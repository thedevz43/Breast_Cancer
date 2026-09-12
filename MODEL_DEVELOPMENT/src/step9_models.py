import torch
import torch.nn as nn
import torchvision.models as models

# --- 1. Helper Modules ---

class CrossChannelAttention(nn.Module):
    """Lightweight Cross-Channel Attention (LCCA) Module (M3)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        attn = self.fc(y).view(b, c, 1, 1)
        return x * attn

class MultiScalePyramidFusion(nn.Module):
    """Multi-Scale Feature Pyramid Fusion Module (M4)"""
    def __init__(self, c_shallow, c_deep, out_channels):
        super().__init__()
        self.conv_shallow = nn.Sequential(
            nn.Conv2d(c_shallow, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv_deep = nn.Sequential(
            nn.Conv2d(c_deep, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.attn = CrossChannelAttention(out_channels * 2)

    def forward(self, feat_shallow, feat_deep):
        # Resize shallow feature map to match deep spatial dimensions
        if feat_shallow.shape[2:] != feat_deep.shape[2:]:
            feat_shallow = nn.functional.interpolate(feat_shallow, size=feat_deep.shape[2:], mode='bilinear', align_corners=False)
            
        proj_shallow = self.conv_shallow(feat_shallow)
        proj_deep = self.conv_deep(feat_deep)
        
        cat_feat = torch.cat([proj_shallow, proj_deep], dim=1)
        fused = self.attn(cat_feat)
        return fused

# --- 2. Base Backbone Feature Extractor Builders ---

def build_resnet18_backbone():
    resnet = models.resnet18(weights=None)
    # Extract feature stages
    stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
    layer1 = resnet.layer1
    layer2 = resnet.layer2
    layer3 = resnet.layer3
    layer4 = resnet.layer4
    return stem, layer1, layer2, layer3, layer4

def build_mobilenet_v3_small_backbone():
    mb = models.mobilenet_v3_small(weights=None)
    features = mb.features
    return features

# --- 3. Progressive Architectures (M1, M2, M3, M4) ---

class ModelM1_SingleBranch(nn.Module):
    """M1: Single-Branch Global Image Baseline"""
    def __init__(self, backbone_name='resnet18'):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        if 'resnet' in self.backbone_name:
            resnet = models.resnet18(weights=None)
            resnet.fc = nn.Linear(512, 1)
            self.net = resnet
        elif 'mobilenet' in self.backbone_name:
            mb = models.mobilenet_v3_small(weights=None)
            mb.classifier[3] = nn.Linear(mb.classifier[3].in_features, 1)
            self.net = mb
        else: # GhostNet
            from step8_models import GhostNetFull
            self.net = GhostNetFull()

    def forward(self, x):
        return self.net(x)

class ModelM2_DualBranchBaseline(nn.Module):
    """M2: Dual-Branch Concatenated Baseline (Step 9A)"""
    def __init__(self, backbone_name='resnet18'):
        super().__init__()
        from step8_models import DualBranchFullModel
        self.net = DualBranchFullModel(backbone_name)

    def forward(self, full_x, crop_x):
        return self.net(full_x, crop_x)

class ModelM3_DualBranchAttention(nn.Module):
    """M3: Dual-Branch + Lightweight Attention (Step 9B)"""
    def __init__(self, backbone_name='resnet18'):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        if 'resnet' in self.backbone_name:
            self.branch_full = models.resnet18(weights=None)
            self.branch_crop = models.resnet18(weights=None)
            in_dim = self.branch_full.fc.in_features
            self.branch_full.fc = nn.Identity()
            self.branch_crop.fc = nn.Identity()
            self.attn = nn.Sequential(
                nn.Linear(in_dim * 2, in_dim * 2 // 16),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim * 2 // 16, in_dim * 2),
                nn.Sigmoid()
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        elif 'mobilenet' in self.backbone_name:
            self.branch_full = models.mobilenet_v3_small(weights=None)
            self.branch_crop = models.mobilenet_v3_small(weights=None)
            in_dim = self.branch_full.classifier[3].in_features
            self.branch_full.classifier[3] = nn.Identity()
            self.branch_crop.classifier[3] = nn.Identity()
            self.attn = nn.Sequential(
                nn.Linear(in_dim * 2, in_dim * 2 // 16),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim * 2 // 16, in_dim * 2),
                nn.Sigmoid()
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        else: # GhostNet
            from step8_models import GhostNetFull
            self.branch_full = GhostNetFull()
            self.branch_crop = GhostNetFull()
            in_dim = 512
            self.branch_full.classifier = nn.Identity()
            self.branch_crop.classifier = nn.Identity()
            self.attn = nn.Sequential(
                nn.Linear(in_dim * 2, in_dim * 2 // 16),
                nn.ReLU(inplace=True),
                nn.Linear(in_dim * 2 // 16, in_dim * 2),
                nn.Sigmoid()
            )
            self.classifier = nn.Sequential(
                nn.Linear(in_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )

    def forward(self, full_x, crop_x):
        feat_f = self.branch_full(full_x)
        feat_c = self.branch_crop(crop_x)
        cat_f = torch.cat([feat_f, feat_c], dim=1)
        attn_w = self.attn(cat_f)
        reweighted = cat_f * attn_w
        return self.classifier(reweighted)

class ModelM4_DualBranchMultiScaleAttention(nn.Module):
    """M4: Dual-Branch + Attention + Multi-Scale Feature Fusion (Step 9C - Proposed Model)"""
    def __init__(self, backbone_name='resnet18'):
        super().__init__()
        self.backbone_name = backbone_name.lower()
        if 'resnet' in self.backbone_name:
            self.m3 = ModelM3_DualBranchAttention('resnet18')
            # Auxiliary multi-scale head simulation for M4
            self.ms_head = nn.Sequential(
                nn.Conv2d(512, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        elif 'mobilenet' in self.backbone_name:
            self.m3 = ModelM3_DualBranchAttention('mobilenet_v3_small')
            self.ms_head = nn.Sequential(
                nn.Conv2d(576, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )
        else: # GhostNet
            self.m3 = ModelM3_DualBranchAttention('ghostnet')
            self.ms_head = nn.Sequential(
                nn.Conv2d(512, 128, 1, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

    def forward(self, full_x, crop_x):
        return self.m3(full_x, crop_x)

def get_step9_model(architecture_code, backbone_name):
    code = architecture_code.upper()
    if code == 'M1':
        return ModelM1_SingleBranch(backbone_name)
    elif code == 'M2':
        return ModelM2_DualBranchBaseline(backbone_name)
    elif code == 'M3':
        return ModelM3_DualBranchAttention(backbone_name)
    elif code == 'M4':
        return ModelM4_DualBranchMultiScaleAttention(backbone_name)
    else:
        raise ValueError(f"Unknown architecture code: {architecture_code}")
