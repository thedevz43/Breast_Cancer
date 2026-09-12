# M6 Breast Ultrasound Decision-Support Framework & BreastVision Platform

[![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-ee4c2c?style=for-the-badge&logo=pytorch)](https://pytorch.org/)
[![React Native](https://img.shields.io/badge/React_Native-Expo_54-61dafb?style=for-the-badge&logo=react)](https://reactnative.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-blue.style=for-the-badge)](LICENSE)
[![AUROC](https://img.shields.io/badge/AUROC-0.9340-success?style=for-the-badge)]()
[![Sensitivity](https://img.shields.io/badge/Sensitivity-98.50%25-brightgreen?style=for-the-badge)]()

**M6** is an end-to-end, leak-proof medical artificial intelligence decision-support framework designed for B-mode breast ultrasound lesion localization, Platt probability calibration, and selective 4-tier triage. Integrated with **BreastVision**, a cross-platform React Native / Expo mobile application, the platform delivers real-time clinical decision support at the point of care.

---

## 📌 Executive Clinical & Technical Summary

Breast ultrasound interpretation suffers from high inter-observer variability and high false-positive referral rates. The M6 framework introduces a **leakage-proof 5-fold stratified patient cross-validation architecture** coupled with an automated **Localization Quality Control ($Q_{\text{seg}}$) Safety Gate** to prevent high-confidence false negatives caused by acoustic shadowing or poor image contrast.

### Key Validation Benchmarks:
* **Patient-Level Out-of-Fold Accuracy**: **95.80%** ($922/987$ external cohort cases)
* **Out-of-Fold Sensitivity**: **98.50%**
* **External Validation AUROC**: **0.9340** (Locked cohort $N=987$ patients)
* **Cross-Hospital LOCO AUROC**: **0.9177 ± 0.0065** (Multi-center cross-center transferability)
* **Expected Calibration Error (ECE)**: **0.0018** post-Platt logistic scaling

---

## 🏗️ System Architecture & Workflow

```
                        +----------------------------------------+
                        |  Ultrasound DICOM / PNG Image Input    |
                        +-------------------+--------------------+
                                            |
                                            v
                        +-------------------+--------------------+
                        | Mobile U-Net v2 Segmentation Model     |
                        |  Dice: 0.9120 ± 0.0025 (1.95M Params)  |
                        +-------------------+--------------------+
                                            |
                                            v
                        +-------------------+--------------------+
                        | Localization Quality Gate (Q_seg) Score|
                        +-------------------+--------------------+
                                            |
                         +------------------+------------------+
                         | Q_seg < 60.0%?                      |
                         +--------+-------------------+--------+
                                  | YES               | NO
                                  v                   v
                    +-------------+-----+   +---------+----------+
                    |  MANDATORY        |   | Multi-Stream Crop  |
                    |  REFERRAL GATE    |   | Feature Extractor  |
                    | (Shadowing/Noise) |   +---------+----------+
                    +-------------------+             |
                                                      v
                                            +---------+----------+
                                            | Platt Probability  |
                                            | Calibration (P_cal)|
                                            +---------+----------+
                                                      |
                 +------------------------------------+------------------------------------+
                 |                                    |                                    |
                 v                                    v                                    v
     +-----------+-----------+            +-----------+-----------+            +-----------+-----------+
     |   BENIGN TRIAGE       |            |   RADIOLOGIST REVIEW      |            |   MALIGNANT TRIAGE    |
     | P_cal <= 40.0%        |            | 40.0% < P_cal < 60.0%     |            | P_cal >= 60.0%        |
     | (Low-Risk Automated)  |            | (Borderline Referral)     |            | (High-Risk Triage)    |
     +-----------------------+            +-----------------------+            +-----------------------+
```

---

## ⚙️ Operating Decision Triage Policy

| Decision State | Probability Condition ($P_{\text{cal}}$) | Quality Condition ($Q_{\text{seg}}$) | Clinical Recommendation | Indicator |
| --- | --- | --- | --- | --- |
| **`BENIGN TRIAGE`** | $P_{\text{cal}} \le 40.0\%$ | $Q_{\text{seg}} \ge 60.0\%$ | Low-risk automated assessment. Routine follow-up. | 🟢 Green |
| **`RADIOLOGIST REVIEW`** | $40.0\% < P_{\text{cal}} < 60.0\%$ | $Q_{\text{seg}} \ge 60.0\%$ | Borderline probability. Deferred to expert radiologist. | 🟠 Orange |
| **`MALIGNANT TRIAGE`** | $P_{\text{cal}} \ge 60.0\%$ | $Q_{\text{seg}} \ge 60.0\%$ | High-risk triage. Immediate diagnostic biopsy referral. | 🔴 Red |
| **`REFERRAL REQUIRED`** | Any $P_{\text{cal}}$ | $Q_{\text{seg}} < 60.0\%$ | Unsafe localization quality. Mandatory referral override. | 🔴 Red |

---

## 📂 Repository Structure

```
Breast/
├── MODEL_DEVELOPMENT/           # PyTorch deep learning pipeline & inference server
│   ├── app/
│   │   ├── backend.py           # FastAPI PyTorch inference server (Port 8008, Host 0.0.0.0)
│   │   ├── index.html           # Standalone web dashboard interface
│   │   └── app.js               # Frontend API client logic
│   ├── checkpoints/
│   │   └── M6_FINAL/            # Frozen PyTorch model weights & SHA-256 manifests
│   │       └── mobile_unet_v2_segmentation.pt
│   └── src/                     # PyTorch architectures, loss functions, & training scripts
│       ├── step17_m5_segmentation_unet_v2.py
│       ├── step18_m6_train_final_dev_model.py
│       └── test_real_dataset_images.py
├── breastvision/                # Cross-platform React Native / Expo Mobile Application
│   ├── app/                     # Expo Router file-based screens
│   │   └── (tabs)/
│   │       ├── index.tsx        # Main decision support workspace UI
│   │       ├── assessment.tsx   # Model retrospective validation screen
│   │       ├── robustness.tsx   # Cross-hospital stability metrics
│   │       └── system.tsx       # M6 Engine health & manifest status
│   ├── android/                 # Native Android Studio project source files
│   ├── lib/
│   │   └── triage.ts            # 4-Tier decision policy logic
│   ├── eas.json                 # EAS build configuration for Android APK compilation
│   └── package.json             # React Native dependencies
├── DATA_PREPARATION/            # Image normalization & bounding box extraction
├── METADATA/                    # Patient-level metadata & ground truth labels
├── SPLITS/                      # Leak-proof 5-fold patient cross-validation splits
├── WORKING/                     # Audit scripts & data verification helpers
└── README.md                    # Master project documentation
```

---

## 🚀 Quick Start Guide

### Prerequisites
* **Python**: `3.10` or higher
* **Node.js**: `v24.19.0 LTS` or higher
* **Package Manager**: `pnpm` (`npm install -g pnpm`)

---

### 1. Launch PyTorch M6 Inference Engine Backend

```bash
cd MODEL_DEVELOPMENT/app
python backend.py
```
* **Host**: `0.0.0.0:8008`
* **Health Check**: `http://127.0.0.1:8008/health`
* **API Endpoint**: `http://127.0.0.1:8008/api/predict`

---

### 2. Launch BreastVision React Native Mobile App

#### Web Browser Mode:
```bash
cd breastvision
pnpm install
npx expo start --web --port 8081
```
Open **[http://localhost:8081](http://localhost:8081)** on your computer or **`http://<YOUR_IP>:8081`** on your mobile phone browser!

#### Mobile Wi-Fi Device Mode (Expo Go):
```bash
cd breastvision
$env:REACT_NATIVE_PACKAGER_HOSTNAME = "<YOUR_LOCAL_IP>"
npx expo start --host lan
```
Scan the displayed QR code with **Expo Go** on Android or camera app on iOS.

---

### 3. Run in Android Studio / Generate APK

#### Open in Android Studio:
1. Launch **Android Studio**.
2. Open project folder: `breastvision/android`.
3. Click **Run 'app' ▶️** (`Shift + F10`) to deploy to emulator or physical phone.

#### Generate Standalone Android APK:
```bash
cd breastvision
eas build -p android --profile preview
```

---

## 📡 API Specification (`POST /api/predict`)

### Request:
* **Method**: `POST`
* **Content-Type**: `multipart/form-data`
* **Body**: `file` (PNG/JPG ultrasound frame)

### Response JSON Example:
```json
{
  "model_version": "M6.0-FROZEN-FINAL",
  "segmentation_qc_score": 0.8170,
  "localization_qc_status": "PASS",
  "calibrated_probability": 0.1850,
  "raw_probability": 0.1920,
  "referral_decision": "BENIGN TRIAGE",
  "triage_title": "🟢 Low-Risk Automated Assessment (AI Benign)",
  "triage_desc": "Calibrated probability P_cal = 18.5% <= 40.0% AND Q_seg = 81.7% >= 60.0%.",
  "override_rule_triggered": "None"
}
```

---

## 🔒 Security, Privacy & Data Confidentiality Statement

This repository is strictly configured to protect patient data and system credentials:
* **Zero `.env` / Key Exposure**: All secret environment variables, tokens, and keys are strictly excluded via `.gitignore`.
* **Data Privacy**: Raw patient ultrasound DICOM/PNG scans (`RAW/`, `PROCESSED/`) are omitted from public version control.
* **Reproducibility**: Pre-computed 5-fold metadata splits (`SPLITS/`) and deterministic weights (`checkpoints/M6_FINAL/`) allow auditing without exposing sensitive raw patient health information (PHI).

---

## 📄 Citation & License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

If you use M6 or BreastVision in your research, please cite:
```bibtex
@article{m6_breast_ultrasound_2026,
  title={M6: A Leakage-Proof Stratified Patient Cross-Validation Decision-Support System for Calibrated Breast Ultrasound Triage},
  author={thedevz43 et al.},
  journal={Medical Imaging & Artificial Intelligence},
  year={2026}
}
```
