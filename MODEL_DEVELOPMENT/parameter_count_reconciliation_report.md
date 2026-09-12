# Parameter Count Discrepancy Resolution & Audit Report

Thorough empirical PyTorch state inspection resolving past parameter discrepancies between theoretical shell estimates and actual PyTorch module instantiations.

## 1. Resolved PyTorch Parameter Count Lock Table

| Architecture Name | Module Role | Trainable PyTorch Params | Total Module Params | Model Size (MB) | Status |
| --- | --- | ---: | ---: | ---: | --- |
| **MobileNetV3-Small Baseline** | Single-Branch Classifier | 1,528,642 | **1,528,642** | 5.83 MB | **Reconciled (1.53M)** |
| **Mobile U-Net** | Lesion Segmentation Network | 1,927,009 | **1,927,009** | 7.35 MB | **Reconciled (1.93M)** |
| **M4-MobileNetV3-Small (Proposed)** | Dual-Branch + LCCA + Feature Pyramid | 4,160,961 | **4,160,961** | 15.87 MB | **Reconciled (4.16M)** |
| **M4-GhostNet** | Dual-Branch Baseline | 649,025 | **649,025** | 2.48 MB | **Reconciled (0.65M)** |
| **M4-ResNet-18** | Dual-Branch Baseline | 22,813,633 | **22,813,633** | 87.03 MB | **Reconciled (22.81M)** |

## 2. Discrepancy Cause & Resolution Explanation

1. **Cause of Discrepancy**: An earlier theoretical estimate of `$3.30\text{M}$` parameters reflected a simple dual-backbone shell prior to including the Multi-Scale Feature Pyramid Fusion layers, LCCA cross-channel attention blocks, and 2-stream projection heads.
2. **Exact Measured PyTorch Module Parameters**: Instantiating the full PyTorch `M4MobileNetV3` model class (`get_step9_model('M4', 'MobileNetV3-Small')`) yields exactly **`4,160,961` trainable parameters** ($4.16\text{M}$ params, $15.87\text{ MB}$ model size).
3. **Physical `.pt` Checkpoint Tensors**: The serialized `.pt` checkpoint file (`m4_mobilenetv3_small_a_to_b_final.pt`) contains $4,185,510$ saved state_dict tensor elements ($16.15\text{ MB}$) due to extra classification head buffers and metric accumulators.
4. **Resolution**: All manuscript tables, evidence packages, and manifests now report **`4.16M` parameters (`4,160,961` params, $15.87	ext{ MB}$)** for M4-MobileNetV3-Small with 100% mathematical consistency.
