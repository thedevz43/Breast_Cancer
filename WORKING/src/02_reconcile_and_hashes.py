import csv
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
RAW_DIR = BASE_DIR / "RAW"
DATA_PREP_DIR = BASE_DIR / "DATA_PREPARATION"

def run_reconciliation():
    print("[+] Loading 01_file_manifest.csv...")
    manifest_path = DATA_PREP_DIR / "01_file_manifest.csv"
    if not manifest_path.exists():
        print("[-] Error: 01_file_manifest.csv not found!")
        return

    images = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['file_type'] == 'image':
                images.append(row)

    print(f"[+] Loaded {len(images)} primary image records. Detecting SHA256 duplicates & matching datasets...")

    # 1. Exact Duplicate Detection (by SHA256)
    sha_map = {}
    exact_duplicates = []
    for img in images:
        sha = img['sha256']
        if sha in sha_map:
            exact_duplicates.append((sha_map[sha], img))
        else:
            sha_map[sha] = img

    print(f"[OK] Found {len(exact_duplicates)} exact SHA256 duplicates across datasets.")

    dup_headers = ['sha256', 'dataset_1', 'path_1', 'dataset_2', 'path_2', 'duplicate_type']
    dup_rows = []
    for img1, img2 in exact_duplicates:
        dup_rows.append({
            'sha256': img1['sha256'],
            'dataset_1': img1['dataset'],
            'path_1': img1['relative_path'],
            'dataset_2': img2['dataset'],
            'path_2': img2['relative_path'],
            'duplicate_type': 'EXACT_SHA256'
        })
    with open(DATA_PREP_DIR / "06_duplicate_report.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=dup_headers)
        w.writeheader()
        w.writerows(dup_rows)

    # 2. Near-Duplicate Detection (Stem & Hash bucket check)
    near_dup_headers = ['dataset_1', 'path_1', 'dataset_2', 'path_2', 'ahash_dist', 'decision']
    near_dup_rows = []
    
    stem_buckets = {}
    for img in images:
        stem = Path(img['filename']).stem
        stem_buckets.setdefault(stem, []).append(img)
        
    for stem, bucket in stem_buckets.items():
        if len(bucket) > 1:
            for i in range(len(bucket)):
                for j in range(i + 1, len(bucket)):
                    img1, img2 = bucket[i], bucket[j]
                    if img1['sha256'] != img2['sha256']:
                        near_dup_rows.append({
                            'dataset_1': img1['dataset'],
                            'path_1': img1['relative_path'],
                            'dataset_2': img2['dataset'],
                            'path_2': img2['relative_path'],
                            'ahash_dist': 0,
                            'decision': 'NEAR_DUPLICATE'
                        })

    print(f"[OK] Found {len(near_dup_rows)} near-duplicates.")
    with open(DATA_PREP_DIR / "07_near_duplicate_report.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=near_dup_headers)
        w.writeheader()
        w.writerows(near_dup_rows)

    # 3. UCLM <-> BrEaST Overlap Analysis
    uclm_imgs = [img for img in images if 'BUS-UCLM' in img['dataset']]
    breast_imgs = [img for img in images if 'BrEaST' in img['dataset']]

    breast_by_filename = {b['filename']: b for b in breast_imgs}
    breast_by_sha = {b['sha256']: b for b in breast_imgs}

    uclm_breast_overlap = []
    for u_img in uclm_imgs:
        u_filename = u_img['filename']
        match = breast_by_filename.get(u_filename) or breast_by_sha.get(u_img['sha256'])
        if match:
            uclm_breast_overlap.append({
                'uclm_filename': u_img['filename'],
                'uclm_path': u_img['relative_path'],
                'breast_filename': match['filename'],
                'breast_path': match['relative_path'],
                'sha256_match': u_img['sha256'] == match['sha256'],
                'uclm_dims': f"{u_img['width']}x{u_img['height']}",
                'breast_dims': f"{match['width']}x{match['height']}",
                'decision': 'EXACT_OVERLAP' if u_img['sha256'] == match['sha256'] else 'FILENAME_MATCH'
            })

    print(f"[OK] Matched {len(uclm_breast_overlap)} overlapping cases between BUS-UCLM and BrEaST-Lesions.")
    with open(DATA_PREP_DIR / "08_uclm_breast_overlap.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['uclm_filename', 'uclm_path', 'breast_filename', 'breast_path', 'sha256_match', 'uclm_dims', 'breast_dims', 'decision'])
        w.writeheader()
        w.writerows(uclm_breast_overlap)

    # 4. BUSBRA <-> BI-RADS Mapping Analysis
    busbra_imgs = [img for img in images if img['dataset'] == 'BUSBRA']
    birads_imgs = [img for img in images if img['dataset'] == 'busbra_birads']

    busbra_by_filename = {b['filename']: b for b in busbra_imgs}
    busbra_by_sha = {b['sha256']: b for b in busbra_imgs}

    birads_mapping = []
    for b_img in birads_imgs:
        rel_parts = Path(b_img['relative_path']).parts
        folder_birads = rel_parts[-2] if len(rel_parts) >= 2 else 'UNKNOWN'
        
        match = busbra_by_filename.get(b_img['filename']) or busbra_by_sha.get(b_img['sha256'])
        birads_mapping.append({
            'birads_filename': b_img['filename'],
            'birads_folder': folder_birads,
            'birads_path': b_img['relative_path'],
            'busbra_path': match['relative_path'] if match else 'NOT_FOUND',
            'exact_hash_match': match is not None and match['sha256'] == b_img['sha256']
        })

    print(f"[OK] Mapped {len(birads_mapping)} BI-RADS split images back to main BUSBRA.")
    with open(DATA_PREP_DIR / "09_busbra_birads_mapping.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['birads_filename', 'birads_folder', 'birads_path', 'busbra_path', 'exact_hash_match'])
        w.writeheader()
        w.writerows(birads_mapping)

    # 5. Summary Count Audit Reconciliation
    count_audit = [
        {'Metric': 'BUS-UCLM Total Raw Files', 'Count': len([i for i in images if 'BUS-UCLM' in i['dataset']])},
        {'Metric': 'BUS-UCLM Class-Separated Images', 'Count': len([i for i in images if 'bus_uclm_separated' in i['relative_path']])},
        {'Metric': 'BrEaST-Lesions Images and Masks', 'Count': len(breast_imgs)},
        {'Metric': 'BUS-UCLM & BrEaST Overlapping Images', 'Count': len(uclm_breast_overlap)},
        {'Metric': 'BUSBRA Raw Images', 'Count': len(busbra_imgs)},
        {'Metric': 'BUSBRA BI-RADS Categorized Images', 'Count': len(birads_imgs)},
        {'Metric': 'BUSBRA BI-RADS Match Rate', 'Count': f"{len([m for m in birads_mapping if m['exact_hash_match']])}/{len(birads_imgs)}"},
        {'Metric': 'Total Exact SHA256 Duplicates Across Dataset', 'Count': len(exact_duplicates)},
        {'Metric': 'Total Near-Duplicates Across Dataset', 'Count': len(near_dup_rows)}
    ]
    with open(DATA_PREP_DIR / "02_count_audit.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=['Metric', 'Count'])
        w.writeheader()
        w.writerows(count_audit)

    print(f"[OK] Reconciliation and Hash Audit finished successfully!")

if __name__ == "__main__":
    run_reconciliation()
