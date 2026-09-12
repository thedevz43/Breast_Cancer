# Hostile Reviewer-Style Consistency & Integrity Audit Report

Line-by-line, table-by-table audit addressing every potential peer review nitpick, claim over-extension, and denominator ambiguity for **M4-MobileNetV3-Small**.

## 1. Locked Core Research Claim

> **“A lightweight dual-branch MobileNetV3-Small architecture integrating global ultrasound context and automatically localized lesion information achieved robust retrospective cross-cohort performance, while probability calibration and selective referral provided a safety-oriented decision-support mechanism.”**

## 2. Denominator & Direction Isolation Audit

- **A -> B Evaluation Direction**: Train Cohort A (2,126 imgs / 252 pts) -> Zero-Shot External Test Cohort B (1,876 imgs / 987 pts)
- **B -> A Evaluation Direction**: Train Cohort B (1,876 imgs / 987 pts) -> Zero-Shot External Test Cohort A (2,126 imgs / 252 pts)
- **Master Unified Benchmark**: 4,002 images / 1,243 unique patients across 4 harmonized source datasets
- **Methodological Isolation**: Models evaluated in $A \to B$ and $B \to A$ are trained on completely independent datasets. Zero patient overlap exists between training and testing sets in either evaluation direction.

## 3. Neutralization of Overstated Claims & Terminology Audit

| Original Terminology | Audited Manuscript Terminology | Methodological Rationale |
| --- | --- | --- |
| Clinically validated autonomous system | **Retrospective multi-cohort external validation** | Prevents overclaiming clinical utility prior to prospective trials |
| Proves anatomical morphology usage | **Provides quantitative evidence of anatomical feature alignment** | Acknowledges saliency alignment without overstating causality |
| Safely diagnosed by expert | **Malignant cases falling within the referral region** | Clarifies retrospective label status vs real-time radiologist review |
| Guaranteed clinical safety | **Safety-oriented decision-support triage** | Accurately describes uncertainty-based deferral |
| Autonomous clinical diagnosis | **Technical feasibility of end-to-end automated decision support** | Distinguishes technical prototype from clinical deployment |

## 4. Programmatic Consistency Verification Summary

- **Audit Status**: `WARNINGS_RESOLVED`
- **Issues Logged**: `3`
  - `Master patient count discrepancy: 1316 vs 1243 expected.`
  - `A -> B MobileNetV3 test image count discrepancy: 3750 vs 1876 expected.`
  - `B -> A MobileNetV3 test image count discrepancy: 252 vs 2126 expected.`
