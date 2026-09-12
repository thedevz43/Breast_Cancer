import numpy as np
import pandas as pd
from metrics import calculate_clinical_metrics

def compute_bootstrap_cis_df(y_true, y_prob, patient_ids=None, n_bootstraps=500):
    rng = np.random.RandomState(42)
    boot_metrics = []
    
    if patient_ids is not None:
        unique_pids = np.unique(patient_ids)
        n_pids = len(unique_pids)
        pid_groups = {pid: np.where(patient_ids == pid)[0] for pid in unique_pids}
        
        for b in range(n_bootstraps):
            sampled_pids = rng.choice(unique_pids, size=n_pids, replace=True)
            sampled_indices = np.concatenate([pid_groups[pid] for pid in sampled_pids])
            y_t_b = y_true[sampled_indices]
            y_p_b = y_prob[sampled_indices]
            
            if len(np.unique(y_t_b)) < 2:
                continue
            m = calculate_clinical_metrics(y_t_b, y_p_b, n_bootstraps=1)
            boot_metrics.append(m)
    else:
        n_samples = len(y_true)
        for b in range(n_bootstraps):
            idx = rng.choice(n_samples, size=n_samples, replace=True)
            y_t_b = y_true[idx]
            y_p_b = y_prob[idx]
            if len(np.unique(y_t_b)) < 2:
                continue
            m = calculate_clinical_metrics(y_t_b, y_p_b, n_bootstraps=1)
            boot_metrics.append(m)
            
    df_boot = pd.DataFrame(boot_metrics)
    
    res = {}
    for col in ['auroc', 'auprc', 'sensitivity', 'specificity', 'f1', 'balanced_accuracy', 'mcc', 'npv', 'ppv', 'accuracy']:
        if col in df_boot.columns:
            m_val = df_boot[col].mean()
            l_val = np.percentile(df_boot[col], 2.5)
            u_val = np.percentile(df_boot[col], 97.5)
            res[col] = f"{m_val:.4f} ({l_val:.4f}–{u_val:.4f})"
    return res

def audit_dual_level_performance(df_preds):
    """
    Computes image-level and patient-level performance separately.
    Patient-level performance uses Mean Probability Aggregation across patient views.
    """
    # 1. Image-Level Performance
    y_true_img = df_preds['true_label'].values
    y_prob_img = df_preds['predicted_probability'].values
    pids_img = df_preds['patient_id'].values
    
    res_img = compute_bootstrap_cis_df(y_true_img, y_prob_img, patient_ids=pids_img)
    
    # 2. Patient-Level Performance (Mean Probability Aggregation)
    patient_records = []
    for pid, group in df_preds.groupby('patient_id'):
        patient_records.append({
            'patient_id': pid,
            'true_label': group['true_label'].iloc[0],
            'mean_prob': np.mean(group['predicted_probability'].values)
        })
    df_pat = pd.DataFrame(patient_records)
    
    y_true_pat = df_pat['true_label'].values
    y_prob_pat = df_pat['mean_prob'].values
    pids_pat = df_pat['patient_id'].values
    
    res_pat = compute_bootstrap_cis_df(y_true_pat, y_prob_pat, patient_ids=pids_pat)
    
    table_rows = []
    for metric_name in ['auroc', 'auprc', 'sensitivity', 'specificity', 'f1', 'balanced_accuracy', 'mcc', 'npv', 'ppv']:
        table_rows.append({
            'Metric': metric_name.upper(),
            'Image-Level (Mean, 95% CI)': res_img[metric_name],
            'Patient-Level (Mean, 95% CI)': res_pat[metric_name]
        })
        
    return pd.DataFrame(table_rows)
