import time
import numpy as np

def calculate_clinical_metrics(y_true, y_prob, threshold=0.5, n_bootstraps=500):
    y_true = np.array(y_true, dtype=int)
    y_prob = np.array(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    sens = tp / float(tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / float(tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / float(tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / float(tn + fn) if (tn + fn) > 0 else 0.0
    
    f1 = 2 * (ppv * sens) / (ppv + sens) if (ppv + sens) > 0 else 0.0
    bacc = 0.5 * (sens + spec)
    
    # MCC calculation
    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = float((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    
    # AUROC and AUPRC
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        auprc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
    except Exception:
        auroc, auprc = 0.5, 0.5
        
    # Bootstrapped AUROC 95% CI
    boot_aurocs = []
    rng = np.random.RandomState(42)
    n_samples = len(y_true)
    for _ in range(n_bootstraps):
        boot_idx = rng.randint(0, n_samples, n_samples)
        if len(np.unique(y_true[boot_idx])) > 1:
            try:
                boot_aurocs.append(roc_auc_score(y_true[boot_idx], y_prob[boot_idx]))
            except Exception:
                pass
                
    if boot_aurocs:
        ci_lower = float(np.percentile(boot_aurocs, 2.5))
        ci_upper = float(np.percentile(boot_aurocs, 97.5))
    else:
        ci_lower, ci_upper = auroc, auroc

    return {
        'auroc': round(auroc, 4),
        'auprc': round(auprc, 4),
        'sensitivity': round(sens, 4),
        'specificity': round(spec, 4),
        'ppv': round(ppv, 4),
        'npv': round(npv, 4),
        'f1': round(f1, 4),
        'balanced_accuracy': round(bacc, 4),
        'mcc': round(mcc, 4),
        'ci_95_lower': round(ci_lower, 4),
        'ci_95_upper': round(ci_upper, 4)
    }

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_inference_latency(model_or_predict_fn, sample_input, is_torch=True, n_runs=50):
    start = time.perf_counter()
    for _ in range(n_runs):
        if is_torch:
            _ = model_or_predict_fn(sample_input)
        else:
            _ = model_or_predict_fn(sample_input)
    end = time.perf_counter()
    latency_ms = ((end - start) / float(n_runs)) * 1000.0
    return round(latency_ms, 2)
