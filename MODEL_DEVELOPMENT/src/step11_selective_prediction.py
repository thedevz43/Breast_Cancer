import numpy as np

def evaluate_selective_prediction(y_true, y_prob, tau_lower, tau_upper):
    total_samples = len(y_true)
    
    # Decisions:
    # 0 -> AI Benign (prob <= tau_lower)
    # 1 -> AI Malignant (prob >= tau_upper)
    # -1 -> Refer to Radiologist (tau_lower < prob < tau_upper)
    
    is_benign = (y_prob <= tau_lower)
    is_malignant = (y_prob >= tau_upper)
    is_automated = is_benign | is_malignant
    is_referred = ~is_automated
    
    n_automated = np.sum(is_automated)
    n_referred = np.sum(is_referred)
    coverage = (n_automated / total_samples) * 100.0 if total_samples > 0 else 0.0
    referral_rate = (n_referred / total_samples) * 100.0 if total_samples > 0 else 0.0
    
    if n_automated > 0:
        y_true_auto = y_true[is_automated]
        y_pred_auto = np.where(is_malignant[is_automated], 1, 0)
        
        errors = (y_pred_auto != y_true_auto)
        selective_risk = (np.sum(errors) / n_automated) * 100.0
        accuracy = 100.0 - selective_risk
        
        # Post-referral Sensitivity (Malignant correctly identified out of all automated + referred malignant)
        tp = np.sum((y_pred_auto == 1) & (y_true_auto == 1))
        fn = np.sum((y_pred_auto == 0) & (y_true_auto == 1)) + np.sum((y_true[is_referred] == 1))
        sensitivity = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        
        # Post-referral NPV
        tn = np.sum((y_pred_auto == 0) & (y_true_auto == 0))
        npv = (tn / (tn + fn)) * 100.0 if (tn + fn) > 0 else 0.0
    else:
        selective_risk = 0.0
        accuracy = 100.0
        sensitivity = 100.0
        npv = 100.0
        
    return {
        'tau_lower': round(float(tau_lower), 4),
        'tau_upper': round(float(tau_upper), 4),
        'coverage_pct': round(float(coverage), 2),
        'referral_pct': round(float(referral_rate), 2),
        'selective_risk_pct': round(float(selective_risk), 2),
        'accuracy_pct': round(float(accuracy), 2),
        'sensitivity_pct': round(float(sensitivity), 2),
        'npv_pct': round(float(npv), 2),
        'n_automated': int(n_automated),
        'n_referred': int(n_referred)
    }

def find_clinical_operating_point(y_true, y_prob, target_sensitivity=0.98):
    """Finds optimal tau_lower & tau_upper achieving target_sensitivity on development OOF data"""
    best_res = None
    best_coverage = -1.0
    
    # Grid search over tau_lower in [0.05, 0.40] and tau_upper in [0.60, 0.95]
    for t_low in np.linspace(0.05, 0.40, 36):
        for t_high in np.linspace(0.60, 0.95, 36):
            if t_low >= t_high:
                continue
                
            res = evaluate_selective_prediction(y_true, y_prob, t_low, t_high)
            
            if res['sensitivity_pct'] >= (target_sensitivity * 100.0) and res['coverage_pct'] > best_coverage:
                best_coverage = res['coverage_pct']
                best_res = res
                
    if best_res is None:
        best_res = evaluate_selective_prediction(y_true, y_prob, 0.15, 0.85)
        
    return best_res

def generate_coverage_risk_curve(y_true, y_prob):
    """Generates coverage-risk curve points across target coverage levels (100%, 95%, 90%, 85%, 80%, 70%, 60%)"""
    curve_points = []
    
    # Uncertainty metric: distance to 0.5 decision boundary
    uncertainty = np.abs(y_prob - 0.5)
    sorted_indices = np.argsort(-uncertainty) # Sort by most confident first
    
    target_coverages = [1.00, 0.95, 0.90, 0.85, 0.80, 0.70, 0.60]
    total_samples = len(y_true)
    
    for cov in target_coverages:
        n_keep = int(total_samples * cov)
        keep_idx = sorted_indices[:n_keep]
        
        y_t_sub = y_true[keep_idx]
        y_p_sub = y_prob[keep_idx]
        y_pred_sub = (y_p_sub >= 0.5).astype(int)
        
        errors = np.sum(y_pred_sub != y_t_sub)
        risk = (errors / n_keep) * 100.0 if n_keep > 0 else 0.0
        acc = 100.0 - risk
        
        # Sensitivity on kept cases
        tp = np.sum((y_pred_sub == 1) & (y_t_sub == 1))
        fn = np.sum((y_pred_sub == 0) & (y_t_sub == 1)) + np.sum(y_true[sorted_indices[n_keep:]] == 1)
        sens = (tp / (tp + fn)) * 100.0 if (tp + fn) > 0 else 0.0
        
        curve_points.append({
            'target_coverage_pct': round(cov * 100.0, 1),
            'actual_coverage_pct': round((n_keep / total_samples) * 100.0, 2),
            'selective_risk_pct': round(risk, 2),
            'accuracy_pct': round(acc, 2),
            'post_referral_sensitivity_pct': round(sens, 2),
            'n_automated': n_keep,
            'n_referred': total_samples - n_keep
        })
        
    return curve_points
