import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression

def compute_ece(y_true, y_prob, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        in_bin = (y_prob >= bin_lower) & (y_prob < bin_upper) if i < n_bins - 1 else (y_prob >= bin_lower) & (y_prob <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_acc = np.mean(y_true[in_bin])
            bin_conf = np.mean(y_prob[in_bin])
            ece += (bin_size / total_samples) * np.abs(bin_acc - bin_conf)
            
    return round(float(ece), 4)

def compute_brier_score(y_true, y_prob):
    return round(float(np.mean((y_prob - y_true) ** 2)), 4)

class PlattCalibrator:
    def __init__(self):
        self.model = LogisticRegression(C=1e5, solver='lbfgs')
        
    def fit(self, y_prob, y_true):
        # Transform probabilities to logit scale with eps clipping
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
        self.model.fit(logits, y_true)
        return self
        
    def predict_proba(self, y_prob):
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped)).reshape(-1, 1)
        return self.model.predict_proba(logits)[:, 1]

class IsotonicCalibrator:
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds='clip')
        
    def fit(self, y_prob, y_true):
        self.model.fit(y_prob, y_true)
        return self
        
    def predict_proba(self, y_prob):
        return self.model.predict(y_prob)

class TemperatureCalibrator:
    def __init__(self, temp=1.2):
        self.temp = temp
        
    def fit(self, y_prob, y_true):
        # Simple optimal temperature search
        best_t = 1.0
        best_ece = 999.0
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        
        for t in np.linspace(0.5, 3.0, 50):
            scaled_prob = 1.0 / (1.0 + np.exp(-logits / t))
            ece = compute_ece(y_true, scaled_prob)
            if ece < best_ece:
                best_ece = ece
                best_t = t
        self.temp = best_t
        return self
        
    def predict_proba(self, y_prob):
        eps = 1e-7
        y_prob_clipped = np.clip(y_prob, eps, 1 - eps)
        logits = np.log(y_prob_clipped / (1 - y_prob_clipped))
        return 1.0 / (1.0 + np.exp(-logits / self.temp))
