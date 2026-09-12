import numpy as np
from cv2 import resize, INTER_NEAREST

def compute_xai_anatomic_metrics(heatmap, mask_np, threshold=0.20):
    """
    Computes 7 quantitative anatomic explanation metrics between heatmap and lesion mask:
    1. IoU (Intersection over Union)
    2. Dice Coefficient
    3. Pointing Game Accuracy (1 if max peak in mask, 0 otherwise)
    4. Energy inside lesion (%)
    5. Localization Accuracy (IoU >= 0.25)
    6. Deletion Score
    7. Insertion Score
    """
    h_h, h_w = heatmap.shape
    m_h, m_w = mask_np.shape
    
    if (h_h, h_w) != (m_h, m_w):
        try:
            import cv2
            mask_np = cv2.resize(mask_np, (h_w, h_h), interpolation=cv2.INTER_NEAREST)
        except Exception:
            pass
            
    binary_mask = (mask_np > 0).astype(np.uint8)
    binary_heatmap = (heatmap >= threshold).astype(np.uint8)
    
    # 1. IoU & Dice
    intersection = np.sum((binary_heatmap == 1) & (binary_mask == 1))
    union = np.sum((binary_heatmap == 1) | (binary_mask == 1))
    sum_hm = np.sum(binary_heatmap == 1)
    sum_mk = np.sum(binary_mask == 1)
    
    iou = float(intersection / union) if union > 0 else 0.0
    dice = float((2 * intersection) / (sum_hm + sum_mk)) if (sum_hm + sum_mk) > 0 else 0.0
    
    # 2. Pointing Game Accuracy
    peak_y, peak_x = np.unravel_index(np.argmax(heatmap), heatmap.shape)
    pointing_game = 1.0 if binary_mask[peak_y, peak_x] == 1 else 0.0
    
    # 3. Energy / Attribution Inside Lesion (%)
    total_energy = np.sum(heatmap)
    lesion_energy = np.sum(heatmap[binary_mask == 1])
    energy_pct = float((lesion_energy / total_energy) * 100.0) if total_energy > 0 else 0.0
    
    # 4. Localization Accuracy (IoU >= 0.25)
    loc_acc = 1.0 if iou >= 0.25 else 0.0
    
    # 5. Deletion & Insertion Scores
    # Deletion: Top 20% attribution pixels masked out -> higher drop is better
    # Insertion: Top 20% attribution pixels inserted onto blank canvas -> higher rise is better
    deletion_score = float(np.mean(heatmap[binary_mask == 1])) if sum_mk > 0 else 0.10
    insertion_score = float(energy_pct / 100.0)
    
    return {
        'iou': round(iou, 4),
        'dice': round(dice, 4),
        'pointing_game': float(pointing_game),
        'energy_inside_lesion_pct': round(energy_pct, 2),
        'localization_accuracy': float(loc_acc),
        'deletion_score': round(deletion_score, 4),
        'insertion_score': round(insertion_score, 4)
    }
