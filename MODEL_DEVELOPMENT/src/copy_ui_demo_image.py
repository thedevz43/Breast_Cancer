import shutil
from pathlib import Path

src = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\c910f9bd-dfb3-4831-9300-4ed6d7c26e95\m5_1_clinical_prototype_ui_1788087515218.png")
dst = Path(r"C:\Users\Admin\.gemini\antigravity-ide\brain\c910f9bd-dfb3-4831-9300-4ed6d7c26e95\m5_1_ui_demo.png")

if src.exists():
    shutil.copy(src, dst)
    print(f"[OK] Copied UI demo image to {dst}")
