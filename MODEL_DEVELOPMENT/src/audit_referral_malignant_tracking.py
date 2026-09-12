import numpy as np
import pandas as pd

def audit_referral_malignant_tracking(df_preds, tau_lower=0.15, tau_upper=0.85):
    """
    Computes exact malignant case tracking across referral regions:
    - Total Malignant Cases
    - AI-Automated Malignant Cases (TP + FN in automated region)
    - Referred Malignant Cases (deferred to radiologist)
    - Missed Malignant Cases (high-confidence FN)
    - AI-Only Sensitivity
    - Overall System Post-Referral Sensitivity
    - Selective Risk (%)
    """
    total_samples = len(df_preds)
    total_malignant = np.sum(df_preds['true_label'].values == 1)
    total_benign = np.sum(df_preds['true_label'].values == 0)
    
    y_true = df_preds['true_label'].values
    y_prob = df_preds['predicted_probability'].values
    
    is_benign_pred = (y_prob <= tau_lower)
    is_malignant_pred = (y_prob >= tau_upper)
    is_automated = is_benign_pred | is_malignant_pred
    is_referred = ~is_automated
    
    # 1. Automated Cases Accounting
    y_true_auto = y_true[is_automated]
    y_pred_auto = np.where(is_malignant_pred[is_automated], 1, 0)
    
    tp_auto = np.sum((y_pred_auto == 1) & (y_true_auto == 1))
    fn_auto = np.sum((y_pred_auto == 0) & (y_true_auto == 1)) # Missed Malignant Cases
    fp_auto = np.sum((y_pred_auto == 1) & (y_true_auto == 0))
    tn_auto = np.sum((y_pred_auto == 0) & (y_true_auto == 0))
    
    # 2. Referred Cases Accounting
    y_true_ref = y_true[is_referred]
    referred_malignant = np.sum(y_true_ref == 1)
    referred_benign = np.sum(y_true_ref == 0)
    
    # 3. Metrics Calculation
    ai_only_sensitivity = (tp_auto / (tp_auto + fn_auto)) * 100.0 if (tp_auto + fn_auto) > 0 else 0.0
    
    # Overall System Sensitivity (TP / Total Malignant, assuming human radiologist correctly diagnoses referred cases)
    # Radiologist correctly diagnoses all referred_malignant -> System TP = tp_auto + referred_malignant
    overall_system_sensitivity = ((tp_auto + referred_malignant) / total_malignant) * 100.0 if total_malignant > 0 else 0.0
    
    n_auto = np.sum(is_automated)
    n_ref = np.sum(is_referred)
    
    selective_risk = ((fn_auto + fp_auto) / n_auto) * 100.0 if n_auto > 0 else 0.0
    coverage = (n_auto / total_samples) * 100.0
    referral_rate = (n_ref / total_samples) * 100.0
    
    tracking_summary = {
        'total_samples': int(total_samples),
        'total_malignant_cases': int(total_malignant),
        'total_benign_cases': int(total_benign),
        'automated_cases_count': int(n_auto),
        'automated_coverage_pct': round(coverage, 2),
        'referred_cases_count': int(n_ref),
        'referral_rate_pct': round(referral_rate, 2),
        'referred_malignant_count': int(referred_malignant),
        'referred_malignant_pct_of_total_mal': round((referred_malignant / total_malignant) * 100.0, 2),
        'referred_benign_count': int(referred_benign),
        'missed_malignant_count_high_conf_fn': int(fn_auto),
        'missed_malignant_pct_of_total_mal': round((fn_auto / total_malignant) * 100.0, 2),
        'ai_only_automated_sensitivity_pct': round(ai_only_sensitivity, 2),
        'overall_system_sensitivity_pct': round(overall_system_sensitivity, 2),
        'selective_risk_pct': round(selective_risk, 2)
    }
    
    return tracking_summary
