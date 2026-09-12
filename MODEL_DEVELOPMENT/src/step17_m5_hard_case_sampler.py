import numpy as np
import pandas as pd

def build_hard_case_training_sample(df_train, hard_fn_cases=None, oversample_factor=2.5):
    """
    Constructs hard-case training sampler targeting:
    - Small iso-echoic BI-RADS 4 lesions
    - Low-contrast lesions with indistinct margins
    - Posterior acoustic shadowing
    - M4 high-confidence false negative cases (18 cases)
    """
    df_hard = df_train.copy()
    df_hard['sample_weight'] = 1.0
    
    # 1. Identify small lesions (below 25th percentile area)
    if 'width' in df_hard.columns and 'height' in df_hard.columns:
        area = df_hard['width'] * df_hard['height']
        q25 = area.quantile(0.25)
        df_hard.loc[area <= q25, 'sample_weight'] *= oversample_factor
        
    # 2. Identify high-confidence false negative cases if patient ID matches
    if hard_fn_cases:
        df_hard.loc[df_hard['patient_id'].isin(hard_fn_cases), 'sample_weight'] *= (oversample_factor * 1.5)
        
    weights = df_hard['sample_weight'].values
    weights /= weights.sum()
    
    # Draw sampled indices with replacement based on hard-case weighting
    sample_indices = np.random.choice(len(df_hard), size=len(df_hard), replace=True, p=weights)
    return df_hard.iloc[sample_indices].reset_index(drop=True)
