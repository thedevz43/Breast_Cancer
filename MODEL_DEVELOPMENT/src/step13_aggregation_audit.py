import pandas as pd
import numpy as np
from metrics import calculate_clinical_metrics

def audit_patient_aggregation_rules(df_preds):
    """
    Compares 3 patient-level probability aggregation rules for multi-view patient images:
    1. Mean Probability
    2. Maximum Malignant Probability
    3. Median Probability
    """
    patient_records = []
    
    for p_id, group in df_preds.groupby('patient_id'):
        y_true = group['true_label'].iloc[0]
        probs = group['predicted_probability'].values
        
        prob_mean = np.mean(probs)
        prob_max = np.max(probs)
        prob_median = np.median(probs)
        
        patient_records.append({
            'patient_id': p_id,
            'true_label': y_true,
            'prob_mean': prob_mean,
            'prob_max': prob_max,
            'prob_median': prob_median,
            'n_views': len(probs)
        })
        
    df_patient = pd.DataFrame(patient_records)
    y_true_patient = df_patient['true_label'].values
    
    res_mean = calculate_clinical_metrics(y_true_patient, df_patient['prob_mean'].values, n_bootstraps=1)
    res_max = calculate_clinical_metrics(y_true_patient, df_patient['prob_max'].values, n_bootstraps=1)
    res_median = calculate_clinical_metrics(y_true_patient, df_patient['prob_median'].values, n_bootstraps=1)
    
    rules_summary = [
        {
            'aggregation_rule': 'Mean Probability Across Views (Selected)',
            'auroc': round(res_mean['auroc'], 4),
            'auprc': round(res_mean['auprc'], 4),
            'sensitivity': round(res_mean['sensitivity'], 4),
            'specificity': round(res_mean['specificity'], 4),
            'f1': round(res_mean['f1'], 4),
            'patient_count': len(df_patient)
        },
        {
            'aggregation_rule': 'Maximum Malignant Probability Across Views',
            'auroc': round(res_max['auroc'], 4),
            'auprc': round(res_max['auprc'], 4),
            'sensitivity': round(res_max['sensitivity'], 4),
            'specificity': round(res_max['specificity'], 4),
            'f1': round(res_max['f1'], 4),
            'patient_count': len(df_patient)
        },
        {
            'aggregation_rule': 'Median Probability Across Views',
            'auroc': round(res_median['auroc'], 4),
            'auprc': round(res_median['auprc'], 4),
            'sensitivity': round(res_median['sensitivity'], 4),
            'specificity': round(res_median['specificity'], 4),
            'f1': round(res_median['f1'], 4),
            'patient_count': len(df_patient)
        }
    ]
    
    return pd.DataFrame(rules_summary)
