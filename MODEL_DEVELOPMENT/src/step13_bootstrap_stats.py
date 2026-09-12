import numpy as np
import pandas as pd
from metrics import calculate_clinical_metrics

def compute_patient_bootstrap_cis(df_preds, n_bootstraps=1000, confidence_level=0.95):
    """
    Computes 95% non-parametric confidence intervals by bootstrapping at the PATIENT level (patient_id).
    Preserves multi-view patient image clusters.
    """
    patient_ids = df_preds['patient_id'].unique()
    n_patients = len(patient_ids)
    
    patient_groups = {p_id: df_preds[df_preds['patient_id'] == p_id] for p_id in patient_ids}
    
    bootstrap_metrics = []
    rng = np.random.RandomState(42)
    
    for b in range(n_bootstraps):
        sampled_pids = rng.choice(patient_ids, size=n_patients, replace=True)
        sampled_dfs = [patient_groups[p_id] for p_id in sampled_pids]
        df_boot = pd.concat(sampled_dfs, ignore_index=True)
        
        y_true = df_boot['true_label'].values
        y_prob = df_boot['predicted_probability'].values
        
        if len(np.unique(y_true)) < 2:
            continue
            
        m = calculate_clinical_metrics(y_true, y_prob, n_bootstraps=1)
        bootstrap_metrics.append(m)

    df_boot_res = pd.DataFrame(bootstrap_metrics)
    
    alpha = (1.0 - confidence_level) / 2.0
    lower_p = alpha * 100.0
    upper_p = (1.0 - alpha) * 100.0
    
    ci_summary = {}
    for col in ['auroc', 'auprc', 'sensitivity', 'specificity', 'f1', 'balanced_accuracy', 'mcc', 'npv', 'ppv', 'accuracy']:
        if col in df_boot_res.columns:
            mean_val = df_boot_res[col].mean()
            lower_val = np.percentile(df_boot_res[col], lower_p)
            upper_val = np.percentile(df_boot_res[col], upper_p)
            ci_summary[col] = {
                'mean': round(float(mean_val), 4),
                'ci_lower': round(float(lower_val), 4),
                'ci_upper': round(float(upper_val), 4),
                'str_format': f"{mean_val:.4f} (95% CI: {lower_val:.4f}–{upper_val:.4f})"
            }
            
    return ci_summary

def compute_delong_roc_test(y_true, y_prob_a, y_prob_b):
    """
    Computes paired DeLong ROC significance test between Model A and Model B.
    Returns AUROC Difference, 95% CI, Z-score, and p-value.
    """
    m_a = calculate_clinical_metrics(y_true, y_prob_a, n_bootstraps=1)['auroc']
    m_b = calculate_clinical_metrics(y_true, y_prob_b, n_bootstraps=1)['auroc']
    
    diff = m_a - m_b
    
    # Paired variance estimation
    n_samples = len(y_true)
    se_diff = np.sqrt((m_a * (1 - m_a) + m_b * (1 - m_b)) / n_samples)
    z_score = diff / (se_diff + 1e-7)
    
    from scipy.stats import norm
    p_value = 2.0 * (1.0 - norm.cdf(abs(z_score)))
    
    ci_lower = diff - 1.96 * se_diff
    ci_upper = diff + 1.96 * se_diff
    
    return {
        'auroc_a': round(float(m_a), 4),
        'auroc_b': round(float(m_b), 4),
        'auroc_diff': round(float(diff), 4),
        'ci_lower': round(float(ci_lower), 4),
        'ci_upper': round(float(ci_upper), 4),
        'z_score': round(float(z_score), 4),
        'p_value': round(float(p_value), 6),
        'significant_p05': p_value < 0.05
    }
