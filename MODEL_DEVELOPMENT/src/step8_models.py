import torch
import torch.nn as nn
import torchvision.models as models

class GhostModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, ratio=2, dw_size=3, stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.out_channels = out_channels
        init_channels = int(out_channels / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(in_channels, init_channels, kernel_size, stride, kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, dw_size // 2, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Identity()
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        return torch.cat([x1, x2], dim=1)[:, :self.out_channels, :, :]

class GhostNetFull(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super(GhostNetFull, self).__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.Sequential(
            GhostModule(16, 32, stride=2),
            GhostModule(32, 64, stride=2),
            GhostModule(64, 128, stride=2),
            GhostModule(128, 256, stride=2),
            GhostModule(256, 512, stride=1),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.blocks(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

def build_full_resnet18(in_channels=3):
    model = models.resnet18(weights=None)
    if in_channels != 3:
        model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

def build_full_mobilenet_v3_small(in_channels=3):
    model = models.mobilenet_v3_small(weights=None)
    if in_channels != 3:
        model.features[0][0] = nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, 1)
    return model

class DualBranchFullModel(nn.Module):
    def __init__(self, base_name):
        super().__init__()
        self.base_name = base_name.lower()
        if 'resnet' in self.base_name:
            self.branch_full = build_full_resnet18()
            self.branch_crop = build_full_resnet18()
            in_dim = self.branch_full.fc.in_features
            self.branch_full.fc = nn.Identity()
            self.branch_crop.fc = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(in_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        elif 'mobilenet' in self.base_name:
            self.branch_full = build_full_mobilenet_v3_small()
            self.branch_crop = build_full_mobilenet_v3_small()
            in_dim = self.branch_full.classifier[3].in_features
            self.branch_full.classifier[3] = nn.Identity()
            self.branch_crop.classifier[3] = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(in_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )
        else:
            self.branch_full = GhostNetFull()
            self.branch_crop = GhostNetFull()
            in_dim = 128
            self.branch_full.classifier = nn.Identity()
            self.branch_crop.classifier = nn.Identity()
            self.classifier = nn.Sequential(
                nn.Linear(in_dim * 2, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Linear(256, 1)
            )

    def forward(self, full_x, crop_x):
        feat_f = self.branch_full(full_x)
        feat_c = self.branch_crop(crop_x)
        fused = torch.cat([feat_f, feat_c], dim=1)
        return self.classifier(fused)

def get_ablation_model(model_name, strategy):
    strategy = strategy.lower()
    model_name = model_name.lower()
    
    if strategy in ['full_image', 'lesion_crop']:
        if 'resnet' in model_name:
            return build_full_resnet18(in_channels=3)
        elif 'mobilenet' in model_name:
            return build_full_mobilenet_v3_small(in_channels=3)
        else:
            return GhostNetFull(in_channels=3)
            
    elif strategy == 'full_plus_mask':
        if 'resnet' in model_name:
            return build_full_resnet18(in_channels=4)
        elif 'mobilenet' in model_name:
            return build_full_mobilenet_v3_small(in_channels=4)
        else:
            return GhostNetFull(in_channels=4)
            
    elif strategy == 'full_plus_crop':
        return DualBranchFullModel(model_name)
        
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
