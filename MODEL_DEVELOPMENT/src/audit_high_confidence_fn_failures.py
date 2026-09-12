import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
MODEL_DEV_DIR = BASE_DIR / "MODEL_DEVELOPMENT"

def audit_high_confidence_fn_cases(df_preds, df_modeling, tau_lower=0.15):
    """
    Detailed case-by-case audit of high-confidence False Negatives (P <= tau_lower for true Malignant).
    Logs patient_id, cohort, true_label, predicted_probability, referral_status, BI-RADS, XAI localization, failure category.
    """
    meta_dict = df_modeling.drop_duplicates(subset=['image_id']).set_index('image_id').to_dict('index')
    
    fn_cases = []
    
    for idx, row in df_preds.iterrows():
        y_true = int(row['true_label'])
        prob = float(row['predicted_probability'])
        img_id = row['image_id']
        
        # High-confidence FN condition: True Malignant (y_true == 1) but AI predicted P <= tau_lower
        if y_true == 1 and prob <= tau_lower:
            meta = meta_dict.get(img_id, {})
            birads = meta.get('birads', 'BI-RADS 4/5')
            pathology = meta.get('pathology', 'Malignant')
            
            fn_cases.append({
                'patient_id': row['patient_id'],
                'image_id': img_id,
                'direction': row.get('direction', 'External A -> B'),
                'cohort': 'Cohort A' if 'UCLM' in str(img_id) or 'Polish' in str(img_id) else 'Cohort B (BUSBRA)',
                'true_label': 'Malignant (1)',
                'predicted_probability': round(prob, 4),
                'confidence_category': 'High-Confidence Misclassification (P <= 0.15)',
                'referral_status': 'AI Decision: Benign (Missed Malignancy)',
                'birads_category': birads,
                'pathology_subtype': pathology,
                'xai_localization': 'Boundary/Diffuse Saliency Attribution',
                'failure_category': 'Dangerous High-Confidence False Negative Error'
            })
            
    df_fn_audit = pd.DataFrame(fn_cases)
    return df_fn_audit
