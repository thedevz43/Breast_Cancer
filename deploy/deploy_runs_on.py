#!/usr/bin/env python3
"""
CLI Deployment Tool for runs-on.dev
BreastVision & M6 Platform
"""

import sys
import os
import argparse
import zipfile
import json
import urllib.request
import urllib.error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEPLOY_DIR = os.path.join(BASE_DIR, "deploy")
API_BASE = "https://runs-on.dev"

def check_subdomain(name: str) -> dict:
    url = f"{API_BASE}/api/check?name={name}"
    req = urllib.request.Request(url, headers={"User-Agent": "BreastVision-Deploy/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            data = json.loads(resp.read().decode())
            return data
    except urllib.error.HTTPError as e:
        body = e.read().decode()
        return {"error": e.code, "detail": body}
    except Exception as e:
        return {"error": str(e)}

def package_zip(source_dir: str, output_zip: str):
    print(f"[+] Packaging static site from '{source_dir}' -> '{output_zip}'...")
    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as z:
        for root, _, files in os.walk(source_dir):
            for file in files:
                abs_path = os.path.join(root, file)
                rel_path = os.path.relpath(abs_path, source_dir)
                z.write(abs_path, rel_path)
    size_kb = os.path.getsize(output_zip) / 1024
    print(f"[OK] Created zip archive ({size_kb:.2f} KB).")

def upload_to_runs_on(token: str, zip_path: str):
    url = f"{API_BASE}/api/sites/deploy"
    print(f"[+] Deploying {zip_path} to runs-on.dev static hosting...")
    
    with open(zip_path, "rb") as f:
        file_bytes = f.read()
        
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(b'Content-Disposition: form-data; name="site"; filename="site.zip"\r\n')
    body.extend(b"Content-Type: application/zip\r\n\r\n")
    body.extend(file_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode())
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
        "User-Agent": "BreastVision-Deploy/1.0"
    }
    
    req = urllib.request.Request(url, data=bytes(body), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req) as resp:
            result = json.loads(resp.read().decode())
            print(f"[SUCCESS] Deployed successfully!")
            print(json.dumps(result, indent=2))
            if "url" in result:
                print(f"\n🌐 Live at: {result['url']}")
            return result
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode()
        print(f"[ERROR {e.code}] Failed to deploy: {err_msg}")
        return {"error": e.code, "detail": err_msg}
    except Exception as e:
        print(f"[ERROR] Exception during deployment: {e}")
        return {"error": str(e)}

def main():
    parser = argparse.ArgumentParser(description="BreastVision runs-on.dev deployment CLI")
    parser.add_argument("--check", type=str, default="breastvision", help="Subdomain name to check availability for")
    parser.add_argument("--target", choices=["prototype", "mobile"], default="prototype", help="Frontend to package (prototype or mobile)")
    parser.add_argument("--token", type=str, default=os.environ.get("RUNS_ON_TOKEN", ""), help="runs-on.dev deploy token (rod1-...)")
    parser.add_argument("--deploy", action="store_true", help="Upload the packaged zip to runs-on.dev")
    args = parser.parse_args()

    # 1. Check availability
    print(f"=== Checking availability for '{args.check}.runs-on.dev' ===")
    status = check_subdomain(args.check)
    print(f"Result: {status}")

    # 2. Package
    src = os.path.join(BASE_DIR, "MODEL_DEVELOPMENT", "app") if args.target == "prototype" else os.path.join(BASE_DIR, "breastvision", "dist")
    out_zip = os.path.join(DEPLOY_DIR, f"site-{args.target}.zip")
    package_zip(src, out_zip)

    # 3. Deploy if requested and token provided
    if args.deploy or args.token:
        if not args.token:
            print("\n[!] Please provide a deploy token via --token <ROD_TOKEN> or set $env:RUNS_ON_TOKEN.")
            print("    You can generate one at https://runs-on.dev/manage after signing in.")
            sys.exit(1)
        upload_to_runs_on(args.token, out_zip)
    else:
        print(f"\n[INFO] Archive ready at '{out_zip}'.")
        print("To deploy automatically with a token from https://runs-on.dev/manage:")
        print(f"  python deploy/deploy_runs_on.py --target {args.target} --token <YOUR_TOKEN>")

if __name__ == "__main__":
    main()
