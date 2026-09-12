import numpy as np

def extract_padded_crop_from_mask(image_np, mask_np, margin_pct=0.10):
    """
    Extracts 10% padded bounding crop from predicted binary mask.
    Falls back to center 50% region if mask is empty/invalid.
    """
    h, w, c = image_np.shape
    y_idx, x_idx = np.where(mask_np > 0)
    
    if len(y_idx) == 0 or len(x_idx) == 0:
        # Fallback to center 50% bounding region
        ymin, ymax = int(h * 0.25), int(h * 0.75)
        xmin, xmax = int(w * 0.25), int(w * 0.75)
    else:
        ymin, ymax = y_idx.min(), y_idx.max()
        xmin, xmax = x_idx.min(), x_idx.max()
        
    box_h = ymax - ymin
    box_w = xmax - xmin
    
    pad_h = int(box_h * margin_pct)
    pad_w = int(box_w * margin_pct)
    
    ymin_p = max(0, ymin - pad_h)
    ymax_p = min(h, ymax + pad_h)
    xmin_p = max(0, xmin - pad_w)
    xmax_p = min(w, xmax + pad_w)
    
    crop = image_np[ymin_p:ymax_p, xmin_p:xmax_p, :]
    
    try:
        import cv2
        crop_resized = cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        crop_resized = np.zeros((h, w, c), dtype=np.uint8)
        
    bbox_coords = (xmin_p, ymin_p, xmax_p, ymax_p)
    return crop_resized, bbox_coords
