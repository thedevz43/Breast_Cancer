import os
import csv
import json
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
RAW_DIR = BASE_DIR / "RAW"
DATA_PREP_DIR = BASE_DIR / "DATA_PREPARATION"

def compute_mask_bbox(mask_np):
    y_indices, x_indices = np.where(mask_np > 0)
    if len(x_indices) == 0:
        return None
    x_min, x_max = int(x_indices.min()), int(x_indices.max())
    y_min, y_max = int(y_indices.min()), int(y_indices.max())
    w = x_max - x_min + 1
    h = y_max - y_min + 1
    return [x_min, y_min, w, h]

def compute_bbox_iou(box1, box2):
    # box format: [x, y, w, h]
    if not box1 or not box2:
        return 0.0
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
        
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = w1 * h1
    box2_area = w2 * h2
    iou = intersection_area / float(box1_area + box2_area - intersection_area)
    return round(iou, 4)

def run_mask_bbox_validation():
    print("[+] Loading file manifest and patient mapping...")
    manifest_path = DATA_PREP_DIR / "01_file_manifest.csv"
    meta_path = DATA_PREP_DIR / "05_metadata_mapping.csv"
    
    if not manifest_path.exists():
        print("[-] File manifest not found!")
        return

    images = []
    masks = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['file_type'] == 'image':
                images.append(r)
            elif r['file_type'] == 'mask':
                masks.append(r)

    print(f"[+] Loaded {len(images)} images and {len(masks)} mask files.")
    
    # Load BUSBRA BBOX metadata
    bbox_meta = {}
    if meta_path.exists():
        with open(meta_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r['bbox_raw']:
                    try:
                        bbox_meta[r['image_id']] = json.loads(r['bbox_raw'])
                    except Exception:
                        pass

    img_mask_mappings = []
    mask_qualities = []
    bbox_qualities = []

    # Map BUSBRA Masks
    busbra_masks_dir = RAW_DIR / "BUSBRA" / "BUSBRA" / "Masks"
    
    for img_rec in images:
        dataset = img_rec['dataset']
        rel_path = Path(img_rec['relative_path'])
        filename = img_rec['filename']
        stem = Path(filename).stem
        
        found_masks = []
        
        # BUSBRA dataset mask matching
        if dataset == 'BUSBRA' and 'Images' in str(rel_path):
            mask_file = busbra_masks_dir / filename
            if mask_file.exists():
                found_masks.append(mask_file)
                
        # UCLM & BrEaST dataset mask matching
        elif 'BUS-UCLM' in dataset or 'BrEaST' in dataset:
            parent_dir = RAW_DIR / rel_path.parent
            # Find tumor mask
            tumor_mask = parent_dir / f"{stem}_tumor.png"
            if tumor_mask.exists():
                found_masks.append(tumor_mask)
            # Find other masks (_other1, _other2, etc.)
            for i in range(1, 10):
                other_mask = parent_dir / f"{stem}_other{i}.png"
                if other_mask.exists():
                    found_masks.append(other_mask)

        # Process each found mask
        for m_file in found_masks:
            m_rel_path = str(m_file.relative_to(RAW_DIR)).replace('\\', '/')
            img_width = int(img_rec['width']) if img_rec['width'] else 0
            img_height = int(img_rec['height']) if img_rec['height'] else 0
            
            mask_valid = True
            dimensions_match = True
            is_empty = False
            mask_area = 0
            coverage_ratio = 0.0
            mask_bbox = None
            
            try:
                with Image.open(m_file) as m_img:
                    mw, mh = m_img.size
                    if (img_width > 0 and mw != img_width) or (img_height > 0 and mh != img_height):
                        dimensions_match = False
                        mask_valid = False
                    
                    mask_np = np.array(m_img.convert('L'))
                    mask_pixel_count = np.count_nonzero(mask_np > 0)
                    mask_area = int(mask_pixel_count)
                    total_pixels = mw * mh
                    coverage_ratio = round(mask_area / float(total_pixels), 4) if total_pixels > 0 else 0.0
                    
                    if mask_area == 0:
                        is_empty = True
                        mask_valid = False
                    else:
                        mask_bbox = compute_mask_bbox(mask_np)
                        
            except Exception as e:
                mask_valid = False
                
            img_mask_mappings.append({
                'dataset': dataset,
                'image_filename': filename,
                'image_path': img_rec['relative_path'],
                'mask_filename': m_file.name,
                'mask_path': m_rel_path,
                'mask_type': 'tumor' if 'tumor' in m_file.name else ('other' if 'other' in m_file.name else 'primary')
            })
            
            mask_qualities.append({
                'dataset': dataset,
                'image_filename': filename,
                'mask_filename': m_file.name,
                'dimensions_match': dimensions_match,
                'is_empty': is_empty,
                'mask_area_pixels': mask_area,
                'coverage_ratio': coverage_ratio,
                'mask_bbox': json.dumps(mask_bbox) if mask_bbox else '',
                'mask_valid': mask_valid
            })

            # Check BBOX IoU if BUSBRA
            img_id = stem
            if img_id in bbox_meta and mask_bbox:
                raw_bbox = bbox_meta[img_id]
                iou = compute_bbox_iou(raw_bbox, mask_bbox)
                
                # Check BBOX boundary validity
                x, y, w, h = raw_bbox
                bbox_valid = (x >= 0 and y >= 0 and w > 0 and h > 0 and (x + w) <= img_width and (y + h) <= img_height)
                
                bbox_qualities.append({
                    'dataset': dataset,
                    'image_id': img_id,
                    'raw_bbox': json.dumps(raw_bbox),
                    'mask_bbox': json.dumps(mask_bbox),
                    'iou': iou,
                    'bbox_valid': bbox_valid
                })

    print(f"[OK] Validated {len(mask_qualities)} image-mask pairs and {len(bbox_qualities)} BBOX entries.")
    
    with open(DATA_PREP_DIR / "04_image_mask_mapping.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'image_filename', 'image_path', 'mask_filename', 'mask_path', 'mask_type'])
        w.writeheader()
        w.writerows(img_mask_mappings)

    with open(DATA_PREP_DIR / "12_mask_quality.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'image_filename', 'mask_filename', 'dimensions_match', 'is_empty', 'mask_area_pixels', 'coverage_ratio', 'mask_bbox', 'mask_valid'])
        w.writeheader()
        w.writerows(mask_qualities)

    with open(DATA_PREP_DIR / "13_bbox_quality.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'image_id', 'raw_bbox', 'mask_bbox', 'iou', 'bbox_valid'])
        w.writeheader()
        w.writerows(bbox_qualities)

    print("[OK] Saved image mask mapping, mask quality, and bbox quality reports!")

if __name__ == "__main__":
    run_mask_bbox_validation()
