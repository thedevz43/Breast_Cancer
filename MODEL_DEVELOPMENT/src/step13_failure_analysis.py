import pandas as pd
import numpy as np

def categorize_failure_modes(df_preds, tau_lower=0.15, tau_upper=0.85):
    """
    Categorizes external False Positives and False Negatives:
    - Dangerous High-Confidence Failures (P >= tau_upper for Benign or P <= tau_lower for Malignant)
    - Appropriately Referred Low-Confidence Cases (tau_lower < P < tau_upper)
    """
    failures = []
    
    for idx, row in df_preds.iterrows():
        y_true = int(row['true_label'])
        prob = float(row['predicted_probability'])
        pred_cls = 1 if prob >= 0.5 else 0
        
        # Check if error occurred
        if pred_cls != y_true:
            error_type = 'False Positive (Benign labeled Malignant)' if y_true == 0 else 'False Negative (Malignant labeled Benign)'
            
            # Check referral status
            is_referred = (tau_lower < prob < tau_upper)
            
            if is_referred:
                failure_cat = 'Appropriately Referred Low-Confidence Case (Safety Net Triggered)'
                clinical_impact = 'Safe: Deferral to Human Radiologist Prevents AI Diagnostic Error'
            else:
                failure_cat = 'Dangerous High-Confidence Failure'
                clinical_impact = 'High Risk: Autonomous AI Misclassification'
                
            failures.append({
                'patient_id': row['patient_id'],
                'image_id': row['image_id'],
                'direction': row.get('direction', 'External'),
                'backbone': row.get('backbone', 'MobileNetV3-Small'),
                'true_label': 'Malignant' if y_true == 1 else 'Benign',
                'predicted_probability': prob,
                'predicted_class': 'Malignant' if pred_cls == 1 else 'Benign',
                'error_type': error_type,
                'is_referred': is_referred,
                'failure_category': failure_cat,
                'clinical_impact': clinical_impact
            })
            
    df_fail = pd.DataFrame(failures)
    
    # Summary Table
    if not df_fail.empty:
        summary = df_fail.groupby(['error_type', 'failure_category']).size().reset_index(name='case_count')
        summary['percentage_of_errors'] = round((summary['case_count'] / len(df_fail)) * 100.0, 2)
    else:
        summary = pd.DataFrame()
        
    return df_fail, summary
