import os
import shutil
import hashlib
import csv
from pathlib import Path
from PIL import Image

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
RAW_DIR = BASE_DIR / "RAW"
WORKING_DIR = BASE_DIR / "WORKING"
SRC_DIR = WORKING_DIR / "src"
PROCESSED_DIR = BASE_DIR / "PROCESSED"
METADATA_DIR = BASE_DIR / "METADATA"
SPLITS_DIR = BASE_DIR / "SPLITS"
QC_DIR = BASE_DIR / "QUALITY_CONTROL"
DATA_PREP_DIR = BASE_DIR / "DATA_PREPARATION"

RAW_ITEMS = [
    "BUS-UCLM",
    "BUSBRA",
    "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx",
    "BrEaST-Lesions_USG-images_and_masks-Dec-15-2023",
    "busbra_birads"
]

def setup_directories():
    for d in [RAW_DIR, WORKING_DIR, SRC_DIR, PROCESSED_DIR, METADATA_DIR, SPLITS_DIR, QC_DIR, DATA_PREP_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    
    print("[+] Moving raw dataset folders into RAW/...")
    for item in RAW_ITEMS:
        src = BASE_DIR / item
        dst = RAW_DIR / item
        if src.exists() and not dst.exists():
            print(f"    Moving {src} -> {dst}")
            shutil.move(str(src), str(dst))
        elif dst.exists():
            print(f"    {item} already inside RAW/")

def compute_hashes(filepath):
    md5 = hashlib.md5()
    sha256 = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            md5.update(chunk)
            sha256.update(chunk)
    return md5.hexdigest(), sha256.hexdigest()

def classify_file_type(filename):
    name_lower = filename.lower()
    if any(ext in name_lower for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']):
        if 'mask' in name_lower or 'tumor' in name_lower or 'other' in name_lower:
            return 'mask'
        return 'image'
    elif any(ext in name_lower for ext in ['.csv', '.xlsx', '.xls', '.txt', '.json']):
        return 'metadata'
    return 'other'

def get_image_info(filepath):
    try:
        with Image.open(filepath) as img:
            w, h = img.size
            channels = len(img.getbands())
            return w, h, channels
    except Exception:
        return None, None, None

def create_file_inventory():
    print("[+] Scanning files in RAW/ to generate 01_file_manifest.csv...")
    manifest_path = DATA_PREP_DIR / "01_file_manifest.csv"
    
    headers = [
        'record_id', 'dataset', 'relative_path', 'filename', 
        'file_type', 'extension', 'file_size_bytes', 
        'width', 'height', 'channels', 'md5', 'sha256'
    ]
    
    records = []
    record_id = 1
    
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            file_path = Path(root) / f
            rel_path = file_path.relative_to(RAW_DIR)
            parts = rel_path.parts
            dataset_name = parts[0] if len(parts) > 0 else 'UNKNOWN'
            
            ext = file_path.suffix.lower()
            size = file_path.stat().st_size
            f_type = classify_file_type(f)
            
            w, h, channels = (None, None, None)
            if ext in ['.png', '.jpg', '.jpeg', '.bmp']:
                w, h, channels = get_image_info(file_path)
            
            md5_val, sha256_val = compute_hashes(file_path)
            
            records.append({
                'record_id': record_id,
                'dataset': dataset_name,
                'relative_path': str(rel_path).replace('\\', '/'),
                'filename': f,
                'file_type': f_type,
                'extension': ext,
                'file_size_bytes': size,
                'width': w if w is not None else '',
                'height': h if h is not None else '',
                'channels': channels if channels is not None else '',
                'md5': md5_val,
                'sha256': sha256_val
            })
            record_id += 1
            if record_id % 500 == 0:
                print(f"    Processed {record_id} files...")
                
    with open(manifest_path, 'w', newline='', encoding='utf-8') as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=headers)
        writer.writeheader()
        writer.writerows(records)
        
    print(f"[OK] Inventory complete! Saved {len(records)} entries to {manifest_path}")

if __name__ == "__main__":
    setup_directories()
    create_file_inventory()
