import csv
import yaml
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
DATA_PREP_DIR = BASE_DIR / "DATA_PREPARATION"

def run_assembly():
    print("[+] Assembling MASTER_METADATA.csv...")
    
    # Load manifest
    manifest = {}
    with open(DATA_PREP_DIR / "01_file_manifest.csv", 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['file_type'] == 'image':
                manifest[r['relative_path']] = r
                
    # Load patient mapping
    patient_map = {}
    with open(DATA_PREP_DIR / "03_patient_mapping.csv", 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            patient_map[r['image_id']] = r

    # Load metadata mapping
    meta_map = {}
    with open(DATA_PREP_DIR / "05_metadata_mapping.csv", 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            meta_map[r['image_id']] = r

    # Load image mask mapping
    mask_map = {}
    with open(DATA_PREP_DIR / "04_image_mask_mapping.csv", 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            mask_map[r['image_path']] = r

    # Load image quality
    quality_map = {}
    with open(DATA_PREP_DIR / "14_image_quality.csv", 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            quality_map[r['relative_path']] = r

    # Load artifact audit
    artifact_map = {}
    with open(DATA_PREP_DIR / "15_artifact_audit.csv", 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            artifact_map[r['relative_path']] = r

    # Load exclusions
    exclusions = set()
    with open(DATA_PREP_DIR / "17_exclusion_registry.csv", 'r', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            exclusions.add(r['relative_path'])

    master_records = []
    
    for rel_path, m_info in manifest.items():
        dataset = m_info['dataset']
        filename = m_info['filename']
        stem = Path(filename).stem
        
        # Match image_id
        img_id = stem
        p_info = patient_map.get(img_id, {})
        meta_info = meta_map.get(img_id, {})
        mask_info = mask_map.get(rel_path, {})
        q_info = quality_map.get(rel_path, {})
        art_info = artifact_map.get(rel_path, {})
        
        # Determine cohort
        if 'BUS-UCLM' in dataset or 'BrEaST' in dataset:
            cohort = 'Cohort_A'
            patient_id = p_info.get('patient_id', f"UCLM_P{stem.replace('case','').zfill(3)}")
        else:
            cohort = 'Cohort_B'
            patient_id = p_info.get('patient_id', f"BUSBRA_P{stem}")
            
        excluded = rel_path in exclusions
        
        master_records.append({
            'record_id': m_info['record_id'],
            'dataset': dataset,
            'cohort': cohort,
            'patient_id': patient_id,
            'case_id': p_info.get('case_id', f"C_{stem}"),
            'image_id': img_id,
            'relative_path': rel_path,
            'filename': filename,
            'file_size_bytes': m_info['file_size_bytes'],
            'width': m_info['width'],
            'height': m_info['height'],
            'channels': m_info['channels'],
            'sha256': m_info['sha256'],
            'mask_path': mask_info.get('mask_path', ''),
            'mask_type': mask_info.get('mask_type', ''),
            'pathology_raw': p_info.get('pathology_raw', ''),
            'histology_raw': p_info.get('histology_raw', ''),
            'birads_raw': meta_info.get('birads_raw', ''),
            'device_raw': meta_info.get('device_raw', ''),
            'bbox_raw': meta_info.get('bbox_raw', ''),
            'age': meta_info.get('age', ''),
            'side': meta_info.get('side', ''),
            'lesion_status': p_info.get('lesion_status', 'unknown'),
            'label_binary': p_info.get('label_binary', ''),
            'ground_truth_source': 'histopathology' if p_info.get('pathology_raw') else 'dataset_label',
            'is_corrupt': q_info.get('is_corrupt', 'False'),
            'blur_score': q_info.get('blur_score', ''),
            'artifact_flag': art_info.get('artifact_flag', 'False'),
            'eligible_classification': not excluded and p_info.get('lesion_status') in ['benign', 'malignant', 'normal'],
            'eligible_segmentation': not excluded and bool(mask_info.get('mask_path')),
            'excluded': excluded
        })

    master_path = DATA_PREP_DIR / "18_MASTER_METADATA.csv"
    headers = list(master_records[0].keys()) if master_records else []
    with open(master_path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(master_records)

    print(f"[OK] Master metadata created with {len(master_records)} entries at {master_path}")

    # Generate DATA_DICTIONARY.xlsx
    dict_data = [
        {'Variable': 'record_id', 'Description': 'Unique sequential record ID', 'Type': 'numerical', 'Allowed Values': '1..N', 'Predictor?': 'No (ID)'},
        {'Variable': 'dataset', 'Description': 'Original dataset folder source', 'Type': 'categorical', 'Allowed Values': 'BUS-UCLM, BUSBRA, BrEaST-Lesions, busbra_birads', 'Predictor?': 'No'},
        {'Variable': 'cohort', 'Description': 'Assigned dataset cohort', 'Type': 'categorical', 'Allowed Values': 'Cohort_A (UCLM/BrEaST), Cohort_B (BUSBRA)', 'Predictor?': 'Grouping'},
        {'Variable': 'patient_id', 'Description': 'Unique patient identifier', 'Type': 'categorical', 'Allowed Values': 'UCLM_PXXX, BUSBRA_PXXXX', 'Predictor?': 'Grouping'},
        {'Variable': 'image_id', 'Description': 'Image filename stem', 'Type': 'categorical', 'Allowed Values': 'caseXXX, bus_XXXX-l/r/s', 'Predictor?': 'No'},
        {'Variable': 'relative_path', 'Description': 'Relative path to image from RAW/', 'Type': 'categorical', 'Allowed Values': 'path string', 'Predictor?': 'No'},
        {'Variable': 'sha256', 'Description': 'Cryptographic SHA256 image hash', 'Type': 'categorical', 'Allowed Values': '64-char hex', 'Predictor?': 'No'},
        {'Variable': 'lesion_status', 'Description': 'Standardized class label', 'Type': 'categorical', 'Allowed Values': 'normal, benign, malignant', 'Predictor?': 'Target'},
        {'Variable': 'label_binary', 'Description': 'Binary classification target', 'Type': 'binary', 'Allowed Values': '0 (Benign), 1 (Malignant)', 'Predictor?': 'Target'},
        {'Variable': 'birads_raw', 'Description': 'Raw BI-RADS rating', 'Type': 'categorical', 'Allowed Values': '2, 3, 4, 5', 'Predictor?': 'Pre-diagnostic Predictor'},
        {'Variable': 'pathology_raw', 'Description': 'Raw pathology description', 'Type': 'categorical', 'Allowed Values': 'benign, malignant, etc.', 'Predictor?': 'Ground Truth ONLY (NO)'},
        {'Variable': 'histology_raw', 'Description': 'Raw histology description', 'Type': 'categorical', 'Allowed Values': 'fibroadenoma, ductal carcinoma, etc.', 'Predictor?': 'Ground Truth ONLY (NO)'},
        {'Variable': 'device_raw', 'Description': 'Ultrasound imaging device model', 'Type': 'categorical', 'Allowed Values': 'GE Logiq 7 @10-14MHz, etc.', 'Predictor?': 'Subgroup Analysis'},
        {'Variable': 'eligible_classification', 'Description': 'Eligible for classification task', 'Type': 'boolean', 'Allowed Values': 'True, False', 'Predictor?': 'Filter'},
        {'Variable': 'eligible_segmentation', 'Description': 'Eligible for segmentation task', 'Type': 'boolean', 'Allowed Values': 'True, False', 'Predictor?': 'Filter'}
    ]
    df_dict = pd.DataFrame(dict_data)
    dict_excel_path = DATA_PREP_DIR / "19_DATA_DICTIONARY.xlsx"
    df_dict.to_excel(dict_excel_path, index=False)
    print(f"[OK] Data dictionary saved to {dict_excel_path}")

    # Generate 20_COHORT_DEFINITION.yaml
    cohort_def = {
        'dataset_version': 'v1.0',
        'cohort_A': {
            'name': 'UCLM_BrEaST_Unified',
            'description': 'Merged BUS-UCLM and BrEaST-Lesions datasets eliminating duplicate image counts',
            'unique_cases': 256,
            'source_datasets': ['BUS-UCLM', 'BrEaST-Lesions-USG-images_and_masks']
        },
        'cohort_B': {
            'name': 'BUSBRA_Full',
            'description': 'BUSBRA dataset with histology, BI-RADS, bounding boxes, device metadata, and masks',
            'unique_views': 1876,
            'source_datasets': ['BUSBRA', 'busbra_birads']
        },
        'exclusion_policy': 'Exclusions are strictly based on corrupt file structure or missing labels, never model performance.',
        'patient_grouping_variable': 'patient_id'
    }
    yaml_path = DATA_PREP_DIR / "20_COHORT_DEFINITION.yaml"
    with open(yaml_path, 'w', encoding='utf-8') as f:
        yaml.dump(cohort_def, f, default_flow_style=False)
    print(f"[OK] Cohort definitions saved to {yaml_path}")

if __name__ == "__main__":
    run_assembly()
