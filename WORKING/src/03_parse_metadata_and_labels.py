import csv
import pandas as pd
from pathlib import Path

BASE_DIR = Path(r"c:\Users\Admin\Desktop\Breast")
RAW_DIR = BASE_DIR / "RAW"
DATA_PREP_DIR = BASE_DIR / "DATA_PREPARATION"

def parse_busbra_csv():
    busbra_csv = RAW_DIR / "BUSBRA" / "BUSBRA" / "bus_data.csv"
    if not busbra_csv.exists():
        print("[-] Error: BUSBRA bus_data.csv not found!")
        return []
    
    records = []
    with open(busbra_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            case_num = r.get('Case', '')
            image_id = r.get('ID', '')
            histology = r.get('Histology', '').strip()
            pathology = r.get('Pathology', '').strip()
            birads = r.get('BIRADS', '').strip()
            side = r.get('Side', '').strip()
            device = r.get('Device', '').strip()
            bbox = r.get('BBOX', '').strip()
            
            # Map binary label
            if pathology.lower() == 'malignant':
                label_binary = 1
                lesion_status = 'malignant'
            elif pathology.lower() == 'benign':
                label_binary = 0
                lesion_status = 'benign'
            else:
                label_binary = ''
                lesion_status = pathology.lower()
                
            records.append({
                'dataset': 'BUSBRA',
                'image_id': image_id,
                'patient_id': f"BUSBRA_P{case_num.zfill(4)}",
                'case_id': f"BUSBRA_C{case_num.zfill(4)}",
                'side': side,
                'pathology_raw': pathology,
                'histology_raw': histology,
                'birads_raw': birads,
                'device_raw': device,
                'bbox_raw': bbox,
                'label_binary': label_binary,
                'lesion_status': lesion_status,
                'ground_truth_source': 'histopathology' if pathology else 'dataset_label',
                'ground_truth_quality': 'high' if pathology else 'medium'
            })
    return records

def parse_breast_excel():
    excel_path = RAW_DIR / "BrEaST-Lesions-USG-clinical-data-Dec-15-2023.xlsx"
    if not excel_path.exists():
        print("[-] Warning: BrEaST clinical Excel file not found!")
        return []
    
    records = []
    try:
        df = pd.read_excel(excel_path)
        print(f"[+] Successfully loaded BrEaST Excel clinical data: {len(df)} rows.")
        for idx, row in df.iterrows():
            # Get case ID or filename
            case_id = str(row.get('Case', row.get('ID', f'case{idx+1:03d}'))).strip()
            pathology = str(row.get('Pathology', row.get('Classification', ''))).strip()
            histology = str(row.get('Histology', '')).strip()
            birads = str(row.get('BI-RADS', row.get('BIRADS', ''))).strip()
            age = row.get('Age', '')
            
            path_lower = pathology.lower()
            if 'malign' in path_lower:
                label_binary = 1
                lesion_status = 'malignant'
            elif 'benign' in path_lower:
                label_binary = 0
                lesion_status = 'benign'
            elif 'normal' in path_lower:
                label_binary = ''
                lesion_status = 'normal'
            else:
                label_binary = ''
                lesion_status = path_lower
                
            records.append({
                'dataset': 'BrEaST-Lesions',
                'image_id': case_id if case_id.startswith('case') else f"case{int(case_id):03d}",
                'patient_id': f"UCLM_P{str(case_id).replace('case','').zfill(3)}",
                'case_id': f"UCLM_C{str(case_id).replace('case','').zfill(3)}",
                'pathology_raw': pathology,
                'histology_raw': histology,
                'birads_raw': birads,
                'age': age,
                'label_binary': label_binary,
                'lesion_status': lesion_status,
                'ground_truth_source': 'clinical_excel',
                'ground_truth_quality': 'high'
            })
    except Exception as e:
        print(f"[-] Error reading BrEaST Excel file: {e}")
        
    return records

def run_parsing():
    print("[+] Parsing metadata for BUSBRA and BrEaST-Lesions datasets...")
    busbra_records = parse_busbra_csv()
    breast_records = parse_breast_excel()
    
    all_records = busbra_records + breast_records
    print(f"[OK] Extracted metadata for {len(all_records)} total items ({len(busbra_records)} BUSBRA, {len(breast_records)} BrEaST).")
    
    # 1. Output 03_patient_mapping.csv
    patient_headers = ['dataset', 'image_id', 'patient_id', 'case_id', 'side', 'lesion_status', 'label_binary']
    with open(DATA_PREP_DIR / "03_patient_mapping.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=patient_headers, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_records)
        
    # 2. Output 05_metadata_mapping.csv
    meta_headers = ['dataset', 'image_id', 'patient_id', 'case_id', 'side', 'device_raw', 'bbox_raw', 'age', 'birads_raw']
    with open(DATA_PREP_DIR / "05_metadata_mapping.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=meta_headers, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_records)

    # 3. Output 10_label_mapping.csv
    label_headers = ['dataset', 'image_id', 'pathology_raw', 'histology_raw', 'lesion_status', 'label_binary']
    with open(DATA_PREP_DIR / "10_label_mapping.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=label_headers, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_records)

    # 4. Output 11_ground_truth_registry.csv
    gt_headers = ['dataset', 'image_id', 'patient_id', 'lesion_status', 'label_binary', 'ground_truth_source', 'ground_truth_quality']
    with open(DATA_PREP_DIR / "11_ground_truth_registry.csv", 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=gt_headers, extrasaction='ignore')
        w.writeheader()
        w.writerows(all_records)

    print("[OK] Saved patient mapping, metadata mapping, label mapping, and ground truth registry!")

if __name__ == "__main__":
    run_parsing()
