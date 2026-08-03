import subprocess
import sys
import os

# Vibe Transfer for SD WebUI Forge - install dependencies
REQUIRED_PACKAGES = ["torch", "numpy", "Pillow", "safetensors"]

for pkg in REQUIRED_PACKAGES:
    try:
        __import__(pkg.replace("-", "_"))
        print(f"[VibeTransfer] {pkg} already installed")
    except ImportError:
        print(f"[VibeTransfer] Installing {pkg}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

print("[VibeTransfer] All dependencies satisfied")