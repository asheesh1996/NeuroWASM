"""
Download pre-built UltraFace-320 ONNX model for lightweight face detection.
Source: ONNX Model Zoo / Ultra-Light-Fast-Generic-Face-Detector
"""
import urllib.request
import os

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/body_analysis/"
    "ultraface/models/version-RFB-320.onnx"
)
OUTPUT_PATH = "/workspace/ultraface_320.onnx"


def download():
    if os.path.exists(OUTPUT_PATH):
        print(f"UltraFace model already exists at {OUTPUT_PATH}")
        return

    print(f"Downloading UltraFace-320 ONNX from:\n  {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, OUTPUT_PATH)
    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"✅ Downloaded UltraFace-320: {size_mb:.1f} MB → {OUTPUT_PATH}")


if __name__ == "__main__":
    download()
