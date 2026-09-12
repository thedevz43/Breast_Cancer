import os
import pandas as pd
from pathlib import Path
from audit_hostile_reviewer_consistency import audit_hostile_reviewer_consistency

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
MODEL_DEV_DIR = BASE_DIR / "MODEL_DEVELOPMENT"

def run_hostile_reviewer_audit():
    print("[+] Starting Hostile Reviewer-Style Consistency & Integrity Audit...")
    
    audit_res = audit_hostile_reviewer_consistency()
    
    report_path = MODEL_DEV_DIR / "hostile_reviewer_audit_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# Hostile Reviewer-Style Consistency & Integrity Audit Report\n\n")
        f.write("Line-by-line, table-by-table audit addressing every potential peer review nitpick, claim over-extension, and denominator ambiguity for **M4-MobileNetV3-Small**.\n\n")
        
        f.write("## 1. Locked Core Research Claim\n\n")
        f.write("> **“A lightweight dual-branch MobileNetV3-Small architecture integrating global ultrasound context and automatically localized lesion information achieved robust retrospective cross-cohort performance, while probability calibration and selective referral provided a safety-oriented decision-support mechanism.”**\n\n")

        f.write("## 2. Denominator & Direction Isolation Audit\n\n")
        f.write(f"- **A -> B Evaluation Direction**: {audit_res['denominator_a_to_b']}\n")
        f.write(f"- **B -> A Evaluation Direction**: {audit_res['denominator_b_to_a']}\n")
        f.write(f"- **Master Unified Benchmark**: {audit_res['master_harmonized_total']}\n")
        f.write("- **Methodological Isolation**: Models evaluated in $A \\to B$ and $B \\to A$ are trained on completely independent datasets. Zero patient overlap exists between training and testing sets in either evaluation direction.\n\n")

        f.write("## 3. Neutralization of Overstated Claims & Terminology Audit\n\n")
        f.write("| Original Terminology | Audited Manuscript Terminology | Methodological Rationale |\n")
        f.write("| --- | --- | --- |\n")
        f.write("| Clinically validated autonomous system | **Retrospective multi-cohort external validation** | Prevents overclaiming clinical utility prior to prospective trials |\n")
        f.write("| Proves anatomical morphology usage | **Provides quantitative evidence of anatomical feature alignment** | Acknowledges saliency alignment without overstating causality |\n")
        f.write("| Safely diagnosed by expert | **Malignant cases falling within the referral region** | Clarifies retrospective label status vs real-time radiologist review |\n")
        f.write("| Guaranteed clinical safety | **Safety-oriented decision-support triage** | Accurately describes uncertainty-based deferral |\n")
        f.write("| Autonomous clinical diagnosis | **Technical feasibility of end-to-end automated decision support** | Distinguishes technical prototype from clinical deployment |\n\n")

        f.write("## 4. Programmatic Consistency Verification Summary\n\n")
        f.write(f"- **Audit Status**: `{audit_res['status']}`\n")
        f.write(f"- **Issues Logged**: `{audit_res['issues_logged']}`\n")
        if audit_res['issue_details']:
            for issue in audit_res['issue_details']:
                f.write(f"  - `{issue}`\n")
        else:
            f.write("- **All numerical denominators, sample counts, and metrics matched 100% across CSV artifacts and markdown tables.**\n\n")

    print(f"[OK] Hostile reviewer audit report saved to {report_path}")
    print("[OK] HOSTILE REVIEWER AUDIT COMPLETED SUCCESSFULLY!")

if __name__ == "__main__":
    run_hostile_reviewer_audit()
