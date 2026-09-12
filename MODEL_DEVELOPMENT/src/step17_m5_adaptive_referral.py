import numpy as np

def evaluate_adaptive_m5_referral(y_true, y_prob_calib, q_seg_scores, tau_lower=0.15, tau_upper=0.85, qc_threshold=0.60):
    """
    Evaluates Adaptive Referral Policy combining probability thresholds and localization QC score:
    Referral Decision = f(P_cal, Q_seg, Uncertainty)
    """
    n_total = len(y_true)
    
    # Condition 1: High confidence probability AND Good localization confidence
    is_benign = (y_prob_calib <= tau_lower) & (q_seg_scores >= qc_threshold)
    is_malignant = (y_prob_calib >= tau_upper) & (q_seg_scores >= qc_threshold)
    
    # Condition 2: Borderline probability OR Poor localization confidence
    is_referral = ~(is_benign | is_malignant)
    
    n_auto = np.sum(is_benign | is_malignant)
    n_referral = np.sum(is_referral)
    
    coverage_pct = round((n_auto / n_total) * 100.0, 2)
    referral_pct = round((n_referral / n_total) * 100.0, 2)
    
    # Evaluate selective risk on automated cases
    auto_indices = np.where(~is_referral)[0]
    if len(auto_indices) > 0:
        auto_targets = y_true[auto_indices]
        auto_preds = (y_prob_calib[auto_indices] >= 0.50).astype(int)
        errors = np.sum(auto_targets != auto_preds)
        selective_risk_pct = round((errors / len(auto_indices)) * 100.0, 2)
    else:
        selective_risk_pct = 0.0
        
    return {
        'total_cases': n_total,
        'automated_coverage_pct': coverage_pct,
        'referral_rate_pct': referral_pct,
        'selective_risk_pct': selective_risk_pct,
        'referred_cases_count': int(n_referral)
    }
