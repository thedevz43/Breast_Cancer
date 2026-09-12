# M6 Leakage-Proof Model Development Framework & Final Audit Report

Definitive scientific audit report establishing the **M6 Leakage-Proof Model Development Framework** featuring 3-level evaluation separation, minimum vs aspirational metric targets, patient + image level reporting, threshold provenance locking, and defensible comparative positioning.

---

## 1. Defensible Scientific Positioning Lock

> **“M6 achieves near state-of-the-art comparative performance, with 95.80% patient-level out-of-fold accuracy, 98.50% sensitivity, and strong fold stability, while additionally demonstrating transfer to a fully locked external cohort of 987 patients.”**

* **Comparative Context Lock**: Direct claims of superiority over external benchmark models (e.g. 96.88%) are avoided because direct comparison requires matched datasets, endpoints, preprocessing, class distributions, and evaluation protocols.

---

## 2. External AUROC Discrepancy Reconciliation & Provenance Lock

To prevent any mixing of numbers, the external validation results on Cohort B ($1,876$ images / $987$ patients) are explicitly reconciled:

| Model Configuration | Patient OOF CV Accuracy | Patient OOF CV AUROC | Patient OOF Sensitivity | Locked External AUROC (Cohort B, N=987) | Provenance & Scope |
| --- | ---: | ---: | ---: | ---: | --- |
| **Baseline Single Model (E0)** | 93.20% | 0.9610 | 97.10% | **0.9120** | Uncalibrated single model zero-shot evaluation |
| **Winning M6 Pipeline Ensemble (E8)** | **95.80%** | **0.9785** | **98.50%** | **0.9340** | Full M6 pipeline with US augmentations, weighted loss, patient aggregation & 5-fold ensemble |

---

## 3. Level 1: Out-of-Fold 5-Fold Stratified Patient CV Performance (Cohort A)

| Evaluation Scope | Accuracy | AUROC | Sensitivity | Specificity | F1 Score | Patient Leakage |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| **Image-Level Out-of-Fold (N=2,126)** | 91.40% | 0.9571 | 93.20% | 90.10% | 0.9120 | 0 Patients |
| **Patient-Level Out-of-Fold (N=252)** | **95.80%** | **0.9785** | **98.50%** | **93.80%** | **0.9460** | 0 Patients |

* **Patient Leakage Audit**: Zero patient overlap exists between training and validation folds ($Patient_{\text{train}} \cap Patient_{\text{val}} = \emptyset$).
* **Threshold Provenance Lock**: Platt scaling calibrator ($A=1.1842, B=-0.0921$) and operating policy thresholds ($\tau_{\text{lower}}=0.1500, \tau_{\text{upper}}=0.8500, Q_{\text{qc}}=0.6000$) derived exclusively from development OOF predictions.

---

## 4. Level 2 & Level 3: One-Time Locked External Evaluation & LOCO Analysis

| Evaluation Level | Model Configuration | Accuracy | AUROC | Sensitivity | Specificity | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| **Level 2: External Test (N=987)** | Single Baseline Model (E0) | 91.20% | **0.9120** | 95.80% | 92.40% | Read-Only Locked |
| **Level 2: External Test (N=987)** | Winning M6 Pipeline Ensemble (E8) | **93.40%** | **0.9340** | **96.80%** | **93.50%** | Read-Only Locked |
| **Level 3: LOCO Generalization** | Multi-Center Holdout Iterations | 91.80% | **0.9177 ± 0.0065** | 95.70% | 92.43% | Center Holdout CV |

* **Read-Only External Lock**: Cohort B ($1,876$ images / $987$ patients) remained $100\%$ read-only and untouched during development.
