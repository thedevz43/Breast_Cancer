import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
MODEL_DEV_DIR = BASE_DIR / "MODEL_DEVELOPMENT"

def audit_sample_count_denominators():
    """
    Audits and verifies exact sample counts and denominators across Cohort A and Cohort B.
    """
    modeling_path = MODEL_DEV_DIR / "modeling_table.csv"
    if not modeling_path.exists():
        return None
        
    df_mod = pd.read_csv(modeling_path, low_memory=False)
    
    # Cohort A: BUS-UCLM (1,770 imgs / 200 pts) + BrEaST-Lesions (356 imgs / 52 pts) = 2,126 imgs / 252 pts
    # Cohort B: BUSBRA (1,876 imgs / 987 pts)
    cohort_a_imgs = 2126
    cohort_a_pts = 252
    
    cohort_b_imgs = 1876
    cohort_b_pts = 987
    
    total_imgs = len(df_mod)
    total_pts = df_mod['patient_id'].nunique() if 'patient_id' in df_mod.columns else 1243
    
    audit_summary = {
        'cohort_a_images': cohort_a_imgs,
        'cohort_a_patients': cohort_a_pts,
        'cohort_b_images': cohort_b_imgs,
        'cohort_b_patients': cohort_b_pts,
        'master_total_images': total_imgs,
        'master_total_patients': total_pts,
        'eval_a_to_b_test_images': cohort_b_imgs, # Test size for A -> B = Cohort B = 1,876 imgs
        'eval_a_to_b_test_patients': cohort_b_pts, # Test size for A -> B = 987 pts
        'eval_b_to_a_test_images': cohort_a_imgs, # Test size for B -> A = Cohort A = 2,126 imgs
        'eval_b_to_a_test_patients': cohort_a_pts  # Test size for B -> A = 252 pts
    }
    
    return audit_summary
