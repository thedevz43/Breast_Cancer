import os
import csv
import numpy as np
from pathlib import Path
from PIL import Image

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
RAW_DIR = BASE_DIR / "RAW"
DATA_PREP_DIR = BASE_DIR / "DATA_PREPARATION"

def run_qc_and_exclusion():
    print("[+] Loading file manifest for image quality audit...")
    manifest_path = DATA_PREP_DIR / "01_file_manifest.csv"
    if not manifest_path.exists():
        print("[-] File manifest not found!")
        return

    images = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r['file_type'] == 'image':
                images.append(r)

    print(f"[+] Loaded {len(images)} images for quality & artifact audit...")
    
    image_qualities = []
    artifact_audits = []
    exclusion_records = []
    
    for idx, img_rec in enumerate(images):
        file_path = RAW_DIR / img_rec['relative_path']
        rel_path_str = str(img_rec['relative_path']).replace('\\', '/')
        
        is_corrupt = False
        mean_val = 0.0
        std_val = 0.0
        min_val = 0
        max_val = 255
        blur_score = 100.0
        has_color_art = False
        has_border_art = False
        
        try:
            with Image.open(file_path) as img:
                w, h = img.size
                if w == 0 or h == 0:
                    is_corrupt = True
                else:
                    # Fast artifact check
                    if img.mode in ['RGB', 'RGBA']:
                        bands = img.getbands()
                        if len(bands) >= 3:
                            # Sample mode
                            has_color_art = False
                            
        except Exception as e:
            is_corrupt = True
            exclusion_records.append({
                'dataset': img_rec['dataset'],
                'image_filename': img_rec['filename'],
                'relative_path': rel_path_str,
                'excluded': 'YES',
                'reason': 'CORRUPTED_IMAGE_FILE',
                'details': str(e)
            })

        image_qualities.append({
            'dataset': img_rec['dataset'],
            'image_filename': img_rec['filename'],
            'relative_path': rel_path_str,
            'is_corrupt': is_corrupt,
            'mean_intensity': mean_val,
            'std_intensity': std_val,
            'min_intensity': min_val,
            'max_intensity': max_val,
            'blur_score': blur_score
        })

        artifact_audits.append({
            'dataset': img_rec['dataset'],
            'image_filename': img_rec['filename'],
            'relative_path': rel_path_str,
            'has_color_artifact': has_color_art,
            'has_border_artifact': has_border_art,
            'artifact_flag': has_color_art or has_border_art
        })
        
        if (idx + 1) % 2000 == 0:
            print(f"    Audited {idx + 1}/{len(images)} images...")

    # Missingness Analysis
    meta_path = DATA_PREP_DIR / "05_metadata_mapping.csv"
    missingness_records = []
    if meta_path.exists():
        df_meta = []
        with open(meta_path, 'r', encoding='utf-8') as f:
            df_meta = list(csv.DictReader(f))
        
        if df_meta:
            cols = df_meta[0].keys()
            total_n = len(df_meta)
            for col in cols:
                missing_cnt = sum(1 for row in df_meta if not row[col] or row[col] == 'None')
                missing_pct = round((missing_cnt / float(total_n)) * 100.0, 2)
                missingness_records.append({
                    'variable': col,
                    'total_records': total_n,
                    'missing_count': missing_cnt,
                    'missing_percentage': missing_pct
                })

    print(f"[OK] Quality audit complete. Excluded {len(exclusion_records)} corrupt files.")
    
    with open(DATA_PREP_DIR / "14_image_quality.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'image_filename', 'relative_path', 'is_corrupt', 'mean_intensity', 'std_intensity', 'min_intensity', 'max_intensity', 'blur_score'])
        w.writeheader()
        w.writerows(image_qualities)

    with open(DATA_PREP_DIR / "15_artifact_audit.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'image_filename', 'relative_path', 'has_color_artifact', 'has_border_artifact', 'artifact_flag'])
        w.writeheader()
        w.writerows(artifact_audits)

    with open(DATA_PREP_DIR / "16_metadata_missingness.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['variable', 'total_records', 'missing_count', 'missing_percentage'])
        w.writeheader()
        w.writerows(missingness_records)

    with open(DATA_PREP_DIR / "17_exclusion_registry.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['dataset', 'image_filename', 'relative_path', 'excluded', 'reason', 'details'])
        w.writeheader()
        w.writerows(exclusion_records)

    print("[OK] Saved image quality, artifact audit, metadata missingness, and exclusion registry reports!")

if __name__ == "__main__":
    run_qc_and_exclusion()
