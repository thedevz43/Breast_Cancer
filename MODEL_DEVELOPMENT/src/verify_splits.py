import os
import csv
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
DATA_PREP_DIR = BASE_DIR / "DATA_PREPARATION"
SPLITS_DIR = BASE_DIR / "SPLITS"
MODEL_DEV_DIR = BASE_DIR / "MODEL_DEVELOPMENT"
SRC_DIR = MODEL_DEV_DIR / "src"

def setup_directories():
    for d in [MODEL_DEV_DIR, SRC_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    print("[OK] Created MODEL_DEVELOPMENT directory structure.")

def build_modeling_table_and_verify():
    setup_directories()
    
    master_path = DATA_PREP_DIR / "18_MASTER_METADATA.csv"
    if not master_path.exists():
        print("[-] Error: 18_MASTER_METADATA.csv not found!")
        return

    df = pd.read_csv(master_path, low_memory=False)
    print(f"[+] Total records in Master Metadata: {len(df)}")
    
    # Filter for primary binary task: Benign vs Malignant only
    # Filter lesion_status in ['benign', 'malignant']
    df['lesion_status_clean'] = df['lesion_status'].astype(str).str.lower()
    df_binary = df[df['lesion_status_clean'].isin(['benign', 'malignant'])].copy()
    
    # Ensure eligible_classification is True
    is_eligible = df_binary['eligible_classification'].astype(str).str.upper() == 'TRUE'
    df_binary = df_binary[is_eligible].copy()
    
    print(f"[+] Primary Binary Modeling Dataset (Benign vs Malignant): {len(df_binary)} records.")
    
    # Load frozen 5-fold CV split lookup tables
    split_a = pd.read_csv(SPLITS_DIR / "cohort_A_5fold.csv") if (SPLITS_DIR / "cohort_A_5fold.csv").exists() else None
    split_b = pd.read_csv(SPLITS_DIR / "cohort_B_5fold.csv") if (SPLITS_DIR / "cohort_B_5fold.csv").exists() else None
    split_pooled = pd.read_csv(SPLITS_DIR / "pooled_5fold.csv") if (SPLITS_DIR / "pooled_5fold.csv").exists() else None
    
    # Map fold numbers
    a_fold_map = dict(zip(split_a['relative_path'], split_a['fold'])) if split_a is not None else {}
    b_fold_map = dict(zip(split_b['relative_path'], split_b['fold'])) if split_b is not None else {}
    pooled_fold_map = dict(zip(split_pooled['relative_path'], split_pooled['fold'])) if split_pooled is not None else {}

    df_binary['fold_cohort_a'] = df_binary['relative_path'].map(a_fold_map).fillna(-1).astype(int)
    df_binary['fold_cohort_b'] = df_binary['relative_path'].map(b_fold_map).fillna(-1).astype(int)
    df_binary['fold_pooled'] = df_binary['relative_path'].map(pooled_fold_map).fillna(-1).astype(int)
    
    # External test splits
    df_binary['split_a_to_b'] = df_binary['cohort'].map({'Cohort_A': 'train', 'Cohort_B': 'external_test'})
    df_binary['split_b_to_a'] = df_binary['cohort'].map({'Cohort_B': 'train', 'Cohort_A': 'external_test'})

    # Save modeling table
    modeling_table_path = MODEL_DEV_DIR / "modeling_table.csv"
    df_binary.to_csv(modeling_table_path, index=False)
    print(f"[OK] Saved primary modeling table ({len(df_binary)} rows) to {modeling_table_path}")

    # Re-Verify Frozen Splits
    print("[+] Re-verifying frozen splits permanently...")
    verifications = []
    
    # Verification 1: Cohort A Patient Leakage
    df_a = df_binary[df_binary['cohort'] == 'Cohort_A']
    a_folds = [set(df_a[df_a['fold_cohort_a'] == f]['patient_id']) for f in range(1, 6)]
    a_leak = sum(len(a_folds[i].intersection(a_folds[j])) for i in range(5) for j in range(i+1, 5))
    verifications.append({'Check': 'Cohort A Patient Cross-Fold Leakage', 'Overlapping_Patients': a_leak, 'Status': 'PASSED' if a_leak == 0 else 'FAILED'})

    # Verification 2: Cohort B Patient Leakage
    df_b = df_binary[df_binary['cohort'] == 'Cohort_B']
    b_folds = [set(df_b[df_b['fold_cohort_b'] == f]['patient_id']) for f in range(1, 6)]
    b_leak = sum(len(b_folds[i].intersection(b_folds[j])) for i in range(5) for j in range(i+1, 5))
    verifications.append({'Check': 'Cohort B Patient Cross-Fold Leakage', 'Overlapping_Patients': b_leak, 'Status': 'PASSED' if b_leak == 0 else 'FAILED'})

    # Verification 3: Pooled Patient Leakage
    p_folds = [set(df_binary[df_binary['fold_pooled'] == f]['patient_id']) for f in range(1, 6)]
    p_leak = sum(len(p_folds[i].intersection(p_folds[j])) for i in range(5) for j in range(i+1, 5))
    verifications.append({'Check': 'Pooled Patient Cross-Fold Leakage', 'Overlapping_Patients': p_leak, 'Status': 'PASSED' if p_leak == 0 else 'FAILED'})

    # Verification 4: External Test Patient Overlap
    ext_overlap = len(set(df_a['patient_id']).intersection(set(df_b['patient_id'])))
    verifications.append({'Check': 'External Test Patient Overlap (Cohort A vs Cohort B)', 'Overlapping_Patients': ext_overlap, 'Status': 'PASSED' if ext_overlap == 0 else 'FAILED'})

    # Verification 5: Exact SHA256 Leakage
    hash_overlap = len(set(df_a['sha256']).intersection(set(df_b['sha256'])))
    verifications.append({'Check': 'External Test SHA256 Image Leakage', 'Overlapping_Patients': hash_overlap, 'Status': 'PASSED' if hash_overlap == 0 else 'FAILED'})

    report_path = MODEL_DEV_DIR / "split_verification_report.csv"
    pd.DataFrame(verifications).to_csv(report_path, index=False)
    print(f"[OK] Split verification complete! Report saved to {report_path}")

if __name__ == "__main__":
    build_modeling_table_and_verify()
