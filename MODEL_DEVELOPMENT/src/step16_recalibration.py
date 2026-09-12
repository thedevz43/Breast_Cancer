import numpy as np
import pandas as pd
from step11_calibration import compute_ece, compute_brier_score, PlattCalibrator
from step11_selective_prediction import evaluate_selective_prediction, find_clinical_operating_point

def recalibrate_automated_pipeline(df_dev_preds):
    """
    Recalibrates probability scaling and optimizes referral thresholds on development data for the automated pipeline.
    """
    y_true = df_dev_preds['true_label'].values
    y_prob_raw = df_dev_preds['predicted_probability'].values
    
    # Fit Platt calibrator on automated crop predictions
    calibrator = PlattCalibrator().fit(y_prob_raw, y_true)
    y_prob_calib = calibrator.predict_proba(y_prob_raw)
    
    ece_raw = compute_ece(y_true, y_prob_raw)
    ece_calib = compute_ece(y_true, y_prob_calib)
    brier_calib = compute_brier_score(y_true, y_prob_calib)
    
    op_res = find_clinical_operating_point(y_true, y_prob_calib, target_sensitivity=0.98)
    
    recalib_summary = {
        'ece_raw': round(ece_raw, 4),
        'ece_calibrated': round(ece_calib, 4),
        'brier_score': round(brier_calib, 4),
        'tau_lower': op_res['tau_lower'],
        'tau_upper': op_res['tau_upper'],
        'coverage_pct': op_res['coverage_pct'],
        'referral_pct': op_res['referral_pct'],
        'selective_risk_pct': op_res['selective_risk_pct'],
        'post_referral_sensitivity_pct': op_res['sensitivity_pct'],
        'post_referral_npv_pct': op_res['npv_pct']
    }
    
    return calibrator, recalib_summary
