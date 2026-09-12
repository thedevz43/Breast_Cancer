import os
import pandas as pd
import numpy as np
from pathlib import Path

from audit_patient_vs_image_metrics import audit_dual_level_performance
from audit_referral_malignant_tracking import audit_referral_malignant_tracking
from audit_high_confidence_fn_failures import audit_high_confidence_fn_cases

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
MODEL_DEV_DIR = BASE_DIR / "MODEL_DEVELOPMENT"

def format_markdown_table(df_summary):
    cols = df_summary.columns.tolist()
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    rows = []
    for _, row in df_summary.iterrows():
        r_str = "| " + " | ".join(str(val) for val in row.values) + " |"
        rows.append(r_str)
    return "\n".join([header, sep] + rows)

def run_verification_audit():
    print("[+] Starting Manuscript-Grade Verification Audit & Clinical Translation Readiness...")
    
    ext_path = MODEL_DEV_DIR / "step10_external_predictions.csv"
    dev_path = MODEL_DEV_DIR / "step9_progressive_ablation_predictions.csv"
    modeling_path = MODEL_DEV_DIR / "modeling_table.csv"
    
    if not ext_path.exists() or not dev_path.exists() or not modeling_path.exists():
        print("[-] Error: Missing prediction files or modeling table!")
        return

    df_ext_all = pd.read_csv(ext_path)
    df_dev_all = pd.read_csv(dev_path)
    df_modeling = pd.read_csv(modeling_path, low_memory=False)

    df_ext_mb = df_ext_all[df_ext_all['backbone'] == 'MobileNetV3-Small'].reset_index(drop=True)
    df_dev_m4 = df_dev_all[(df_dev_all['architecture_code'] == 'M4') & (df_dev_all['backbone'] == 'MobileNetV3-Small')].reset_index(drop=True)

    # 1. Dual Image-Level vs Patient-Level Audit
    print("  [>] Auditing Dual Image-Level vs Patient-Level Performance (Patient-Cluster 95% CIs)...")
    df_dual_perf = audit_dual_level_performance(df_ext_mb)

    # 2. Detailed Referral Malignant Tracking Audit
    print("  [>] Auditing Detailed Referral Malignant Case Tracking...")
    ref_track = audit_referral_malignant_tracking(df_ext_mb, tau_lower=0.15, tau_upper=0.85)

    # 3. Deep Case-by-Case Audit of High-Confidence False Negatives
    print("  [>] Performing Case-by-Case Audit of High-Confidence False Negatives...")
    df_fn_cases = audit_high_confidence_fn_cases(df_ext_mb, df_modeling, tau_lower=0.15)
    fn_csv_path = MODEL_DEV_DIR / "high_confidence_fn_case_audit.csv"
    df_fn_cases.to_csv(fn_csv_path, index=False)
    print(f"[OK] High-confidence FN case audit saved to {fn_csv_path}")

    # Generate Summary Markdown Audit Report
    report_path = MODEL_DEV_DIR / "manuscript_verification_audit_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Manuscript-Grade Verification Audit & Clinical Translation Report\n\n")
        f.write("Rigorous statistical audit, dual image/patient performance breakdown, exact malignant tracking, and high-confidence false negative audit for **M4-MobileNetV3-Small**.\n\n")
        
        f.write("## 1. Image-Level vs Patient-Level Performance Breakdown\n\n")
        f.write(format_markdown_table(df_dual_perf))
        f.write("\n\n* **Patient-Cluster Rigor**: Patient-level performance was calculated using frozen **Mean Probability Aggregation across patient views** with 95% CIs computed via patient-cluster non-parametric bootstrapping.\n\n")

        f.write("## 2. Detailed Referral System & Malignant Tracking Audit\n\n")
        f.write(f"- **Total External Test Samples**: `{ref_track['total_samples']}` images (`{ref_track['total_malignant_cases']}` malignant, `{ref_track['total_benign_cases']}` benign)\n")
        f.write(f"- **Automated AI Coverage**: `{ref_track['automated_coverage_pct']}%` (`{ref_track['automated_cases_count']}` cases handled automatically)\n")
        f.write(f"- **Selective Referral Rate**: `{ref_track['referral_rate_pct']}%` (`{ref_track['referred_cases_count']}` cases deferred to human radiologists)\n")
        f.write(f"- **Referred Malignant Patients**: `{ref_track['referred_malignant_count']}` cases (**`{ref_track['referred_malignant_pct_of_total_mal']}%` of total malignant cohort**)\n")
        f.write(f"- **Missed Malignant Cases (High-Confidence FN)**: `{ref_track['missed_malignant_count_high_conf_fn']}` cases (**`{ref_track['missed_malignant_pct_of_total_mal']}%` of total malignant cohort**)\n")
        f.write(f"- **AI-Only Automated Sensitivity**: `{ref_track['ai_only_automated_sensitivity_pct']}%`\n")
        f.write(f"- **Overall System Post-Referral Sensitivity**: **`{ref_track['overall_system_sensitivity_pct']}%`** (assuming expert radiologist review of referred cases)\n")
        f.write(f"- **Selective Error Risk on Automated Cases**: **`{ref_track['selective_risk_pct']}%`**\n\n")

        f.write("## 3. High-Confidence False Negative Failure Audit\n\n")
        if not df_fn_cases.empty:
            f.write(format_markdown_table(df_fn_cases.head(10)))
            f.write(f"\n\n* **Total High-Confidence FNs**: `{len(df_fn_cases)}` cases out of `{ref_track['total_malignant_cases']}` malignant samples (`{ref_track['missed_malignant_pct_of_total_mal']}%`). Complete case log exported to [high_confidence_fn_case_audit.csv](file:///c:/Users/Admin/Desktop/Breast/MODEL_DEVELOPMENT/high_confidence_fn_case_audit.csv).\n\n")

        f.write("## 4. DeLong Wording & Methodological Refinement\n\n")
        f.write("1. **DeLong Statistical Comparison**: M4-MobileNetV3 vs M4-ResNet18 ($p = 0.5592$). Wording updated to **'no statistically significant difference detected'** rather than claiming non-inferiority/equivalence without a prespecified equivalence margin.\n")
        f.write("2. **Crop Stream Dependency**: Explicitly framed as `Ultrasound + Expert-Provided Lesion Localization → Calibrated Classifier → AI / Referral Triage`.\n")
        f.write("3. **Translational Roadmap**: Automatic lesion detection/segmentation (e.g. MobileUNet / YOLOv8-seg) identified as the immediate next translational experiment to convert expert-provided crops into end-to-end automated crops.\n")

    print(f"[OK] Manuscript verification audit report saved to {report_path}")
    print("[OK] MANUSCRIPT VERIFICATION AUDIT COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_verification_audit()
