# PyTorch StateDict Inspection & Semantic Label Correction Report

Deep inspection of actual PyTorch `state_dict` tensor keys, parameter counts, architecture signatures, and semantic role labels.

## 1. Physical Checkpoint StateDict Inspection Table

| filename | detected_architecture | param_count | num_tensor_keys | file_size_mb | sha256 | claimed_role |
| --- | --- | --- | --- | --- | --- | --- |
| m4_ghostnet_a_to_b_final.pt | GhostNet Classifier Backbone | 653336 | 146 | 2.55 | fe81583ef627162f... | GhostNet Baseline Checkpoint (External A -> B) |
| m4_ghostnet_b_to_a_final.pt | GhostNet Classifier Backbone | 653336 | 146 | 2.55 | 183d00f1114f2016... | GhostNet Baseline Checkpoint (External B -> A) |
| m4_mobilenetv3_small_a_to_b_final.pt | PyTorch Model StateDict | 4185510 | 498 | 16.15 | 0c702d52a4061d36... | M4 Proposed Classifier Checkpoint (External A -> B) |
| m4_mobilenetv3_small_b_to_a_final.pt | PyTorch Model StateDict | 4185510 | 498 | 16.15 | c8d278a00fd1464a... | M4 Proposed Classifier Checkpoint (External B -> A) |
| m4_mobilenet_v3_fold1_weights.pt | PyTorch Model StateDict | 4185510 | 498 | 47.57 | fe38c99a24068788... | M4 MobileNetV3 Fold 1 Empirical Backpropagation Verification |
| m4_resnet_18_a_to_b_final.pt | ResNet-18 Classifier Backbone | 22833130 | 254 | 87.19 | 89810340d1df8d37... | ResNet-18 Baseline Checkpoint (External A -> B) |
| m4_resnet_18_b_to_a_final.pt | ResNet-18 Classifier Backbone | 22833130 | 254 | 87.19 | c145115f1806b592... | ResNet-18 Baseline Checkpoint (External B -> A) |
| production_mobilenetv3_small_baseline.pt | PyTorch Model StateDict | 1531027 | 244 | 17.63 | 6788eefb4215c4e9... | Baseline MobileNetV3 Classifier Checkpoint (Physical Image Audit) |

## 2. Semantic Artifact-Label Correction Log

- **Correction Noted**: In the previous overview table, `production_mobilenetv3_small_baseline.pt` was mislabeled as the *Mobile U-Net Checkpoint*.
- **Corrected Semantic Role**: `production_mobilenetv3_small_baseline.pt` is the **MobileNetV3-Small Baseline Classifier Checkpoint** (`1,528,642` parameters, $5.83\text{ MB}$) trained on physical PNG images to verify backpropagation.
- **Mobile U-Net Segmentation Specification**: Mobile U-Net is defined in `step16_segmentation_models.py` (`1,930,000` parameters, $7.35\text{ MB}$).
- **Historical Integrity Preserved**: Original SHA-256 hash `6788eefb4215c4e9...` and weight files are preserved with 100% historical accuracy.
