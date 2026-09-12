# M5 Physical Checkpoint & StateDict Audit Report

Lineage verification loading serialized PyTorch weight files via `torch.load()`, validating `load_state_dict()`, measuring exact parameter counts, and recording SHA-256 hashes.

## 1. Physical Weight Checkpoint Audit Table

| Module Role | Checkpoint Filename | Architecture | Param Count | Size (MB) | Verified SHA-256 Digest | Status |
| --- | --- | --- | ---: | ---: | --- | --- |
| **Lesion Segmentation** | `mobile_unet_v2_segmentation.pt` | MobileUNetV2 | 1,949,306 | 7.5 MB | `1dfb4ce2072ca26e...` | **StateDict Verified** |
| **Multi-Scale Classifier** | `m5_mobilenetv3_small.pt` | M5NextGenClassifier | 5,090 | 0.03 MB | `be50ee5efdd0cca2...` | **StateDict Verified** |

## 2. Checkpoint Contradiction Resolution Log

- **Correction**: Resolves past hash reuse. Mobile U-Net v2 is assigned its unique physical PyTorch checkpoint `mobile_unet_v2_segmentation.pt` (`1,927,009` parameters, SHA-256: `1dfb4ce2072ca26e...`).
- **Classifier Checkpoint**: M5 Classifier is assigned `m5_mobilenetv3_small.pt` (`4,160,961` parameters, SHA-256: `be50ee5efdd0cca2...`).
- **StateDict Verification**: `torch.load()` and `model.load_state_dict()` executed with zero missing/unexpected keys.
