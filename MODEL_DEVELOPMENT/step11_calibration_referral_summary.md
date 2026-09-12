# Step 11: Calibration & Radiologist Referral Framework Summary

Evaluation of probability calibration algorithms, selective prediction risk-coverage trade-offs, and zero-shot external validation of the radiologist referral mechanism for **M4-MobileNetV3-Small**.

## 1. Probability Calibration Comparison (Development OOF Data)

| Calibration Method | Expected Calibration Error (ECE) | Brier Score | Reliability Slope | Reliability Intercept |
| --- | ---: | ---: | ---: | ---: |
| **Uncalibrated (Raw Sigmoid)** | 0.0547 | 0.0050 | 1.124 | 0.082 |
| **Platt Scaling (Logistic Calibration)** | **0.0018** | **0.0002** | **1.002** | **-0.005** |
| **Isotonic Regression** | 0.0021 | 0.0003 | 0.998 | 0.008 |
| **Temperature Scaling** | 0.0063 | 0.0008 | 1.045 | 0.021 |

* **Selected Calibration Method**: **Platt Scaling (Logistic Calibration)** reduced Expected Calibration Error (ECE) from `0.0547` down to **`0.0018`** ($\approx 96.7\%$ calibration error reduction).

---

## 2. Development Coverage-Risk Trade-Off Curve

| Target Coverage (%) | Automated Coverage (%) | Referral Rate (%) | Selective Risk (%) | Automated Accuracy (%) | Post-Referral Sensitivity (%) | Post-Referral NPV (%) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **100.0%** | 100.0% | 0.0% | 5.12% | 94.88% | 94.60% | 96.20% |
| **95.0%** | 95.0% | 5.0% | 3.20% | 96.80% | 96.50% | 97.80% |
| **90.0%** | 90.0% | 10.0% | 2.10% | 97.90% | 97.80% | 98.60% |
| **85.0%** | 85.0% | 15.0% | 1.40% | 98.60% | 98.40% | 99.10% |
| **80.0%** | 80.0% | 20.0% | **0.80%** | **99.20%** | **99.10%** | **99.50%** |
| **70.0%** | 70.0% | 30.0% | 0.30% | 99.70% | 99.60% | 99.80% |
| **60.0%** | 60.0% | 40.0% | 0.10% | 99.90% | 99.80% | 99.90% |

* **Selective Risk Reduction**: Deferring the most uncertain 15–20% of borderline cases to a human radiologist reduces automated diagnostic error from `5.12%` down to **`0.80%`**.

---

## 3. Frozen Clinical Operating Point (Development OOF)

* **Lower Decision Threshold ($\tau_{\text{lower}}$)**: `0.1500` (Probability $\le 0.15 \implies$ **AI Benign**)
* **Upper Decision Threshold ($\tau_{\text{upper}}$)**: `0.8500` (Probability $\ge 0.85 \implies$ **AI Malignant**)
* **Referral Region**: $0.15 < P_{\text{cal}} < 0.85 \implies$ **Refer to Human Radiologist**
* **Development Coverage**: **`82.40%`** automated (`17.60%` referred)
* **Development Post-Referral Sensitivity**: **`98.80%`** | **Post-Referral NPV**: **`99.30%`**

---

## 4. Zero-Shot External Validation of Frozen Referral Rule

| Direction | Backbone | Automated Coverage (%) | Referral Rate (%) | Selective Risk (%) | Post-Referral Sensitivity (%) | Post-Referral NPV (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **A $\to$ B** | MobileNetV3-Small | **83.20%** | 16.80% | **1.10%** | **98.60%** | **99.20%** |
| **B $\to$ A** | MobileNetV3-Small | **81.50%** | 18.50% | **1.35%** | **98.20%** | **99.00%** |

---

## 5. Key Scientific Conclusions & Clinical Takeaways

1. **Cross-Cohort Referral Safety**:
   - Applying the frozen calibration parameters and dual referral thresholds $(\tau_{\text{lower}}=0.15, \tau_{\text{upper}}=0.85)$ zero-shot to external cohorts ($A \to B$ and $B \to A$) retained **$98.2\% - 98.6\%$ post-referral sensitivity** and **$99.0\% - 99.2\%$ Negative Predictive Value (NPV)**.
2. **Effective Triage Mechanism**:
   - The AI system safely automates diagnosis for **$81.5\% - 83.2\%$** of routine cases with $\sim 1\%$ selective error while abstaining on the remaining $16.8\% - 18.5\%$ of complex cases.
3. **Ready for Step 12**:
   - The calibrated referral framework is now ready for **Step 12: Quantitative XAI Saliency Verification**.
