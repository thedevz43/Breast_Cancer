# Prospective Clinical Validation Study Protocol & Reader Trial Design

**Title**: Prospective Multi-Center Crossover Reader Study Evaluating a Calibrated Dual-Branch Lightweight AI Decision-Support Pipeline for Breast Ultrasound Triage

---

## 1. Executive Summary & Study Rationale

Following successful retrospective multi-cohort external validation ($A \to B: 0.9035$ AUROC; $B \to A: 0.8935$ AUROC), this protocol defines the prospective clinical validation framework required to establish the clinical safety, diagnostic efficacy, and reader efficiency of the **M4-MobileNetV3 Automated Decision-Support Pipeline** (`Raw Ultrasound → Mobile U-Net → Automated Crop → M4 Classifier → Recalibrated Triage`).

### Core Research Claim Under Validation
> **"AI-assisted radiologist evaluation using the M4 decision-support pipeline improves diagnostic accuracy and reading speed while maintaining non-inferior sensitivity relative to unassisted expert radiologists under a Multi-Reader Multi-Case (MRMC) evaluation framework."**

---

## 2. Multi-Reader Multi-Case (MRMC) Trial Architecture

```text
               Prospective Diagnostic Patients (N = 450)
                                   │
                                   ▼
             ┌──────────────────────────────────────────┐
             │ Eligibility Screening & Consent          │
             │ (Biopsy GT / 2-Year Follow-Up)           │
             └──────────────────────────────────────────┘
                                   │
                                   ▼
             ┌──────────────────────────────────────────┐
             │ Williams Latin Square 2x2 Crossover      │
             └──────────────────────────────────────────┘
                   │                              │
                   ▼                              ▼
      ┌──────────────────────────┐  ┌──────────────────────────┐
      │  Arm 1: Unassisted Read  │  │  Arm 2: AI-Assisted Read │
      │  (Standard B-Mode US)    │  │  (US + M4 + Grad-CAM++)  │
      └──────────────────────────┘  └──────────────────────────┘
                   │                              │
                   ▼                              ▼
             ┌──────────────────────────────────────────┐
             │ 4-Week Washout Period (Case Reshuffling) │
             └──────────────────────────────────────────┘
                   │                              │
                   ▼                              ▼
      ┌──────────────────────────┐  ┌──────────────────────────┐
      │  Arm 2: AI-Assisted Read │  │  Arm 1: Unassisted Read  │
      └──────────────────────────┘  └──────────────────────────┘
                   │                              │
                   └──────────────┬───────────────┘
                                  ▼
             ┌──────────────────────────────────────────┐
             │ Obuchowski-Rockette / Hillis (ORH) MRMC  │
             │ Random-Effects ANOVA Non-Inferiority     │
             └──────────────────────────────────────────┘
```

---

## 3. Statistically Rigorous Primary Endpoint & MRMC Hypothesis

### Primary Endpoint Definition
The primary endpoint is the difference in Multi-Reader Multi-Case (MRMC) composite AUROC between AI-assisted radiologist readings ($\text{AUROC}_{\text{AI+Rad}}$) and unassisted radiologist readings ($\text{AUROC}_{\text{Rad}}$):
$$\Delta \text{AUROC} = \text{AUROC}_{\text{AI+Rad}} - \text{AUROC}_{\text{Rad}}$$

### Non-Inferiority Hypothesis Specification
* **Null Hypothesis ($H_0$)**: AI assistance decreases radiologist diagnostic accuracy by more than the non-inferiority margin $\delta = 0.03$:
  $$H_0: \Delta \text{AUROC} \le -0.03$$
* **Alternative Hypothesis ($H_1$)**: AI-assisted radiologist diagnostic accuracy is non-inferior to unassisted reading:
  $$H_1: \Delta \text{AUROC} > -0.03$$

Rejection of $H_0$ occurs if the lower bound of the two-sided $95\%$ confidence interval for $\Delta \text{AUROC}$ exceeds $-0.03$ ($p < 0.025$, one-sided non-inferiority test).

---

## 4. Multi-Reader Multi-Case (MRMC) Sample Size Derivation

To account for both **case-level variance ($\sigma_C^2$)** and **reader-level variance ($\sigma_R^2$)** under the **Obuchowski-Rockette / Hillis (ORH)** random-effects ANOVA model, sample size was derived using the following explicit assumptions:

### ORH Powering Parameters
* **Target Non-Inferiority Margin**: $\delta = 0.03$
* **Type-I Error Rate**: $\alpha = 0.05$ (two-sided)
* **Statistical Power**: $1 - \beta = 0.80$
* **Reader Panel Size ($R$)**: $12$ radiologists (4 Junior, 4 Senior Generalists, 4 Breast Specialists)
* **Expected Unassisted Baseline AUROC**: $0.8500$
* **Expected AI-Assisted AUROC**: $0.8750$
* **ORH Variance Components**:
  - Case variance component $\text{Var}(C) = 0.0150$
  - Reader variance component $\text{Var}(R) = 0.0050$
  - Reader-by-case interaction $\text{Var}(RC) = 0.0250$
  - Case covariance across modalities $\text{Cov}_1 = 0.0080$
  - Reader covariance across modalities $\text{Cov}_2 = 0.0030$
  - Reader-case covariance $\text{Cov}_3 = 0.0120$

### Required Sample Size
Under these ORH variance parameters, **$N = 450$ prospective patient cases** ($150$ malignant, $300$ benign; 1:2 prevalence ratio) across 3 clinical centers yields **$83.4\%$ statistical power** to reject $H_0$ at $\alpha = 0.05$.

---

## 5. 10-Point Methodological Verification Audit

| Audit Item | Methodological Specification | Operational Implementation |
| --- | --- | --- |
| **1. MRMC Power Calculation** | Obuchowski-Rockette / Hillis (ORH) ANOVA | $N=450$ cases, $R=12$ readers, $\delta=0.03$, Power $= 83.4\%$ |
| **2. Primary Endpoint** | Difference in MRMC-adjusted AUROC | $\Delta \text{AUROC} = \text{AUROC}_{\text{AI+Rad}} - \text{AUROC}_{\text{Rad}}$ |
| **3. Non-Inferiority Margin** | One-sided test with $\delta = 0.03$ | Reject $H_0$ if $95\%\text{ CI lower bound} > -0.03$ |
| **4. MRMC Statistical Model** | Random-effects ANOVA with jackknife variance | Accounts for both case-level and reader-level clustering |
| **5. Crossover Randomization** | Williams $2 \times 2$ Latin square sequence | Randomly assigns reading order to eliminate sequence bias |
| **6. Washout Controls** | 4-week washout with reshuffled case order | Prevents case recall and memory contamination |
| **7. Reader Learning Effects** | Period-by-modality interaction terms | Evaluated in the ORH ANOVA model |
| **8. AI Referral Handling** | Flagged referral cases ($0.15 < P_{\text{cal}} < 0.85$) | Prompts mandatory radiologist review without overriding judgment |
| **9. Reference Standard Timing** | Core-needle biopsy within 30 days / 2-yr stability | Histological verification for all suspicious lesions |
| **10. Prospective Model Freeze** | Locked PyTorch weights & Platt calibrator | **$0$ online retraining**; SHA-256 weight hash verification |

---

## 6. Reader Panel Stratification & Subgroup Analyses

| Reader Stratum | Experience Level | Reader Count | Primary Research Question |
| --- | --- | ---: | --- |
| **Stratum 1: Junior Readers** | Radiology Residents / Fellows ($< 3$ yrs) | 4 | Does AI elevate junior accuracy to senior baseline? |
| **Stratum 2: Senior Generalists** | General Radiologists ($5-10$ yrs) | 4 | Does AI reduce inter-reader variability and reading time? |
| **Stratum 3: Breast Specialists** | Dedicated Breast Subspecialists ($> 10$ yrs) | 4 | Does AI maintain specialist accuracy while speeding workflow? |

---

## 7. Statistical Analysis Plan & Reporting Compliance

1. **Primary Analysis**: Obuchowski-Rockette / Hillis (ORH) random-effects ANOVA model for MRMC ROC data, testing non-inferiority of $\Delta \text{AUROC}$ with jackknife variance estimation.
2. **Secondary Analyses**:
   - Mixed-effects logistic regression for paired sensitivity and specificity comparisons.
   - Paired Student's t-test / linear mixed model for reading latency ($\text{seconds/case}$).
   - Decision Curve Analysis (DCA) quantifying Net Clinical Benefit across threshold probabilities $\tau \in [0.05, 0.50]$.
3. **Reporting Compliance**: Fully compliant with **CONSORT-AI** (Consolidated Standards of Reporting Trials - AI extension) and **STARD-AI** guidelines.
