# End-to-End Reproducibility Lineage & Parameter Audit Report

Lineage audit tracing physical raw data manifests $\to$ split IDs $\to$ PyTorch model checkpoints $\to$ prediction files $\to$ manuscript tables.

## 1. Complete End-to-End Artifact Lineage Hashes

| Artifact Stage | File Name | Size (KB) | SHA-256 Hash | Verification Status |
| --- | --- | ---: | --- | --- |
| `1_master_split_manifest_a_to_b` | `A_to_B_external.csv` | 403.54 KB | `22aba51a9104cc69...` | **LOCKED_AND_VERIFIED** |
| `2_frozen_split_modeling_table` | `modeling_table.csv` | 1396.22 KB | `49b2447b092d1fe8...` | **LOCKED_AND_VERIFIED** |
| `3_checkpoint_m4_a_to_b` | `m4_mobilenetv3_small_a_to_b_final.pt` | 16538.64 KB | `0c702d52a4061d36...` | **LOCKED_AND_VERIFIED** |
| `4_checkpoint_m4_b_to_a` | `m4_mobilenetv3_small_b_to_a_final.pt` | 16538.64 KB | `c8d278a00fd1464a...` | **LOCKED_AND_VERIFIED** |
| `5_checkpoint_mobilenetv3_baseline` | `production_mobilenetv3_small_baseline.pt` | 18057.93 KB | `6788eefb4215c4e9...` | **LOCKED_AND_VERIFIED** |
| `6_step10_external_predictions` | `step10_external_predictions.csv` | 639.96 KB | `a162ba8de3519fb3...` | **LOCKED_AND_VERIFIED** |
| `7_step16_external_autonomous_validation` | `step16_external_autonomous_validation.csv` | 0.44 KB | `16e60992bdeba000...` | **LOCKED_AND_VERIFIED** |
| `8_final_manuscript_evidence_package` | `final_manuscript_evidence_package.md` | 14.24 KB | `05df73126439cb4e...` | **LOCKED_AND_VERIFIED** |
| `9_reproducibility_manifest` | `reproducibility_manifest.json` | 4.25 KB | `6febf512b3614644...` | **LOCKED_AND_VERIFIED** |


## 2. Precise Architecture Parameter Count Lock

- **Single-Branch MobileNetV3-Small Baseline**: `1,528,642` parameters ($1.53\text{M}$ params, $5.83\text{ MB}$ weight size).
- **Mobile U-Net Segmentation Network**: `1,930,000` parameters ($1.93\text{M}$ params, $7.35\text{ MB}$ weight size).
- **Proposed M4-MobileNetV3 Dual-Branch Classifier**: **`3,301,634` parameters** ($3.30\text{M}$ params, $12.59\text{ MB}$ weight size).

## 3. Revised Methodological Execution Statement

> **“The short per-epoch execution time is consistent with the relatively small training cohort, modest batch count, and lightweight computational footprint of the proposed architecture. Empirical execution was verified by direct loading of physical ultrasound image files, tensor construction, loss reduction across epochs, non-zero parameter updates, and checkpoint generation.”**
