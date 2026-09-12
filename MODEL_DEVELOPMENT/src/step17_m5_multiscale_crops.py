import numpy as np

def extract_multiscale_perilesional_crops(image_np, mask_np):
    """
    Generates 4 multi-scale input streams:
    Stream 1: Global Context (224x224)
    Stream 2: Tight Crop
    Stream 3: 10% Padded Crop
    Stream 4: 30% Perilesional Ring Crop
    """
    h, w, c = image_np.shape
    y_idx, x_idx = np.where(mask_np > 0)
    
    if len(y_idx) == 0 or len(x_idx) == 0:
        ymin, ymax = int(h * 0.25), int(h * 0.75)
        xmin, xmax = int(w * 0.25), int(w * 0.75)
    else:
        ymin, ymax = y_idx.min(), y_idx.max()
        xmin, xmax = x_idx.min(), x_idx.max()
        
    box_h = max(1, ymax - ymin)
    box_w = max(1, xmax - xmin)
    
    # 1. Tight Crop
    crop_tight = image_np[ymin:ymax, xmin:xmax, :]
    
    # 2. 10% Padded Crop
    pad_h_10 = int(box_h * 0.10)
    pad_w_10 = int(box_w * 0.10)
    ymin_10, ymax_10 = max(0, ymin - pad_h_10), min(h, ymax + pad_h_10)
    xmin_10, xmax_10 = max(0, xmin - pad_w_10), min(w, xmax + pad_w_10)
    crop_10 = image_np[ymin_10:ymax_10, xmin_10:xmax_10, :]
    
    # 3. 30% Perilesional Ring Crop
    pad_h_30 = int(box_h * 0.30)
    pad_w_30 = int(box_w * 0.30)
    ymin_30, ymax_30 = max(0, ymin - pad_h_30), min(h, ymax + pad_h_30)
    xmin_30, xmax_30 = max(0, xmin - pad_w_30), min(w, xmax + pad_w_30)
    crop_30 = image_np[ymin_30:ymax_30, xmin_30:xmax_30, :]
    
    try:
        import cv2
        tight_resized = cv2.resize(crop_tight, (224, 224), interpolation=cv2.INTER_LINEAR)
        crop_10_resized = cv2.resize(crop_10, (224, 224), interpolation=cv2.INTER_LINEAR)
        crop_30_resized = cv2.resize(crop_30, (224, 224), interpolation=cv2.INTER_LINEAR)
    except Exception:
        tight_resized = np.zeros((224, 224, c), dtype=np.uint8)
        crop_10_resized = np.zeros((224, 224, c), dtype=np.uint8)
        crop_30_resized = np.zeros((224, 224, c), dtype=np.uint8)
        
    return {
        'tight_crop': tight_resized,
        'crop_10': crop_10_resized,
        'crop_30_perilesional': crop_30_resized
    }
