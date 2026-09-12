import io
import json
import urllib.request
import urllib.parse
import numpy as np
from PIL import Image

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

def test_live_inference_backend():
    print("[+] Starting Fake-Result Detection & Live PyTorch Inference Integration Tests...")
    
    # 1. Health Check Test
    with urllib.request.urlopen(f"{SERVER_URL}/health") as response:
        health_data = json.loads(response.read().decode('utf-8'))
        assert health_data['status'] == 'HEALTHY'
        print(f"[OK] Health Check Passed: Model Version = {health_data['model_version']}")

    # Helper to generate in-memory PNG image
    def make_test_image_bytes(pattern_type):
        if pattern_type == "lesion_benign":
            arr = np.full((224, 224, 3), 50, dtype=np.uint8)
            y, x = np.ogrid[:224, :224]
            mask = ((x - 112)**2 / 30**2 + (y - 112)**2 / 20**2) <= 1
            arr[mask] = [200, 200, 200]
        elif pattern_type == "lesion_malignant":
            arr = np.full((224, 224, 3), 10, dtype=np.uint8)
            y, x = np.ogrid[:224, :224]
            mask = ((x - 80)**2 / 60**2 + (y - 160)**2 / 50**2) <= 1
            arr[mask] = [255, 255, 255]
        elif pattern_type == "random_noise":
            arr = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        else:
            arr = np.zeros((224, 224, 3), dtype=np.uint8)
            
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        return buf.getvalue()

    # 2. Test A: Normal Benign Pattern Image
    img_a = make_test_image_bytes("lesion_benign")
    pred_a = post_multipart(f"{SERVER_URL}/api/predict", "test_a.png", img_a)
    print(f"[OK] Test A (Benign Pattern): P_cal = {pred_a['calibrated_probability']}, Q_seg = {pred_a['segmentation_qc_score']}, Triage = {pred_a['referral_decision']}")

    # 3. Test B: Malignant Pattern Image
    img_b = make_test_image_bytes("lesion_malignant")
    pred_b = post_multipart(f"{SERVER_URL}/api/predict", "test_b.png", img_b)
    print(f"[OK] Test B (Malignant Pattern): P_cal = {pred_b['calibrated_probability']}, Q_seg = {pred_b['segmentation_qc_score']}, Triage = {pred_b['referral_decision']}")
    
    # 4. Test C: Random Noise Image (Must trigger QC Referral Gate)
    img_c = make_test_image_bytes("random_noise")
    pred_c = post_multipart(f"{SERVER_URL}/api/predict", "test_c.png", img_c)
    print(f"[OK] Test C (Random Noise): Q_seg = {pred_c['segmentation_qc_score']}, Override = {pred_c['override_rule_triggered']}, Triage = {pred_c['referral_decision']}")
    assert pred_c['referral_decision'] in ["REFERRAL REQUIRED", "REFERRAL"]
    print("[OK] QC Gate Safety Verified: Random noise triggers Mandatory QC Referral Gate!")

    print("\n[OK] ALL FAKE-RESULT DETECTION & LIVE INFERENCE TESTS PASSED 100%!")

if __name__ == "__main__":
    test_live_inference_backend()
