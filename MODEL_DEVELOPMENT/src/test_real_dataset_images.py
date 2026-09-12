import urllib.request
import json
from pathlib import Path

SERVER_URL = "http://127.0.0.1:8008"

def post_multipart(url, filename, file_bytes):
    boundary = '----WebKitFormBoundary7MA4YWxkTrZu0gW'
    body = []
    body.append(f'--{boundary}'.encode('utf-8'))
    body.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode('utf-8'))
    body.append('Content-Type: image/png\r\n'.encode('utf-8'))
    body.append(file_bytes)
    body.append(f'--{boundary}--'.encode('utf-8'))
    body_data = b'\r\n'.join(body)
    
    req = urllib.request.Request(url, data=body_data)
    req.add_header('Content-Type', f'multipart/form-data; boundary={boundary}')
    with urllib.request.urlopen(req) as response:
        res_body = response.read().decode('utf-8')
        return json.loads(res_body)

raw_dir = Path(r"c:\Users\Admin\Desktop\Breast\RAW")
all_files = [p for p in raw_dir.glob("**/*") if p.suffix.lower() in ['.png', '.jpg', '.jpeg']]

benign_files = [p for p in all_files if "ben" in str(p).lower() or "benign" in str(p).lower()][:5]
malignant_files = [p for p in all_files if "mal" in str(p).lower() or "carcinoma" in str(p).lower()][:5]

print("\n--- BENIGN REAL ULTRASOUND SCANS ---")
for f in benign_files:
    with open(f, 'rb') as fp:
        res = post_multipart(f"{SERVER_URL}/api/predict", f.name, fp.read())
    print(f"Benign {f.name}: P_cal = {res['calibrated_probability']*100:.1f}%, Q_seg = {res['segmentation_qc_score']*100:.1f}% ({res['localization_qc_status']}) -> Decision: {res['referral_decision']}")

print("\n--- MALIGNANT REAL ULTRASOUND SCANS ---")
for f in malignant_files:
    with open(f, 'rb') as fp:
        res = post_multipart(f"{SERVER_URL}/api/predict", f.name, fp.read())
    print(f"Malignant {f.name}: P_cal = {res['calibrated_probability']*100:.1f}%, Q_seg = {res['segmentation_qc_score']*100:.1f}% ({res['localization_qc_status']}) -> Decision: {res['referral_decision']}")
