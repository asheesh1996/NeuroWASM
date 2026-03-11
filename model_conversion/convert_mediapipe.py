"""
Convert MediaPipe BlazeFace short-range TFLite model to ONNX format.

The converted ONNX model (blazeface_128.onnx) is consumed by mediapipeDetector.ts.

Expected model I/O after conversion:
  Input  (inputNames[0]): [1, 128, 128, 3]  float32, NHWC, range [-1, 1]
                           Normalize raw pixels as: (pixel - 127.5) / 127.5
  Output (outputNames[0]): [1, 896, 16]  float32  — raw box regressors
  Output (outputNames[1]): [1, 896,  1]  float32  — raw logit scores (pre-sigmoid)

Anchor decoding (MediaPipe convention, fixed_anchor_size=true, scales=128):
  y_center = raw[0] / 128 + anchor.cy   (raw[0] is Y-axis)
  x_center = raw[1] / 128 + anchor.cx   (raw[1] is X-axis)
  h        = exp(raw[2] / 128)
  w        = exp(raw[3] / 128)

Anchor grid (896 anchors, strides [8,16,16,16], 2 anchors per cell):
  stride  8: 16×16 grid × 2 = 512 anchors
  stride 16: 8×8 grid × 2 × 3 layers = 384 anchors

The TypeScript side (mediapipeDetector.ts) performs anchor decoding, sigmoid,
and best-detection selection at runtime — no post-processing is baked into ONNX.

Usage:
  python convert_mediapipe.py [--output /path/blazeface_128.onnx]

Requirements (see requirements.txt):
  tflite2onnx, onnxsim, flatbuffers
"""

import os
import sys
import subprocess
import urllib.request
import argparse

# ---------------------------------------------------------------------------
# Where to find the BlazeFace short-range TFLite model.
# The file is shipped inside the MediaPipe Python package data directory, or
# can be downloaded from the MediaPipe model card on GitHub.
# ---------------------------------------------------------------------------

# Primary: MediaPipe Python package ships the TFLite model locally.
# We discover it by importing mediapipe and looking for the asset.
MEDIAPIPE_MODEL_NAME = "face_detection_short_range.tflite"

# Fallback 1: Original full-precision TFLite from the MediaPipe GitHub repo.
# This is the canonical float32 model used by MediaPipe internally.
MEDIAPIPE_GITHUB_URL = (
    "https://raw.githubusercontent.com/google/mediapipe/master/"
    "mediapipe/modules/face_detection/face_detection_short_range.tflite"
)

# Fallback 2: Google Storage — float32 variant (float16 gives garbage ONNX).
MEDIAPIPE_STORAGE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_detector/blaze_face_short_range/float32/latest/"
    "blaze_face_short_range.tflite"
)

OUTPUT_PATH_DEFAULT = "/workspace/blazeface_128.onnx"
TFLITE_TMP_PATH = "/workspace/blazeface_short_range.tflite"


def _ensure_libgl() -> None:
    """Install libGL if missing (needed for mediapipe on slim Docker images)."""
    try:
        import ctypes
        ctypes.CDLL("libGL.so.1")
    except OSError:
        print("Installing libGL (required by mediapipe)...")
        subprocess.run(
            ["apt-get", "install", "-y", "-q", "--no-install-recommends", "libgl1"],
            check=False,
            capture_output=True,
        )


def find_or_download_tflite() -> str:
    """Return path to the BlazeFace short-range TFLite file."""

    # 1. Try to locate the file bundled inside the mediapipe package.
    _ensure_libgl()
    try:
        import mediapipe as mp
        pkg_dir = os.path.dirname(mp.__file__ or "")  # type: ignore[arg-type]
        for root, _, files in os.walk(pkg_dir):
            for fname in files:
                if fname == MEDIAPIPE_MODEL_NAME:
                    path = os.path.join(root, fname)
                    print(f"Found bundled TFLite at: {path}")
                    return path
        print("⚠️  mediapipe installed but TFLite not found in package — will download")
    except ImportError:
        print("mediapipe not installed, falling back to direct download")

    os.makedirs(os.path.dirname(TFLITE_TMP_PATH), exist_ok=True)

    # 2. Try GitHub (canonical full-precision float32 model).
    for url in [MEDIAPIPE_GITHUB_URL, MEDIAPIPE_STORAGE_URL]:
        try:
            print(f"Downloading BlazeFace TFLite from:\n  {url}")
            urllib.request.urlretrieve(url, TFLITE_TMP_PATH)
            size_kb = os.path.getsize(TFLITE_TMP_PATH) / 1024
            print(f"Downloaded to: {TFLITE_TMP_PATH}  ({size_kb:.1f} KB)")
            return TFLITE_TMP_PATH
        except Exception as e:
            print(f"  ⚠️  Failed ({e}), trying next URL...")

    raise RuntimeError("All BlazeFace TFLite download attempts failed.")


def convert_tflite_to_onnx(tflite_path: str, output_path: str) -> None:
    """Convert TFLite → ONNX using tflite2onnx."""
    try:
        import tflite2onnx
    except ImportError:
        print("ERROR: tflite2onnx not installed. Run: pip install tflite2onnx")
        sys.exit(1)

    print(f"\nConverting {tflite_path} → {output_path} ...")
    tflite2onnx.convert(tflite_path, output_path)
    print(f"✅ tflite2onnx conversion complete: {output_path}")


def simplify_onnx(output_path: str) -> None:
    """Simplify the ONNX graph with onnxsim (optional but recommended)."""
    try:
        import onnx
        from onnxsim import simplify as onnxsim_simplify

        model = onnx.load(output_path)
        model_simplified, ok = onnxsim_simplify(model)
        if ok:
            onnx.save(model_simplified, output_path)
            print(f"✅ onnxsim simplification applied: {output_path}")
        else:
            print("⚠️  onnxsim could not simplify the model (original kept)")
    except ImportError:
        print("onnxsim not installed — skipping simplification (optional)")


def verify_onnx(output_path: str) -> None:
    """Print model input/output info and run onnx.checker."""
    import onnx
    model = onnx.load(output_path)
    onnx.checker.check_model(model)

    graph = model.graph
    print("\n📥 ONNX inputs:")
    for inp in graph.input:
        dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
        print(f"  {inp.name}: {dims}")

    print("📤 ONNX outputs:")
    for out in graph.output:
        dims = [d.dim_value for d in out.type.tensor_type.shape.dim]
        print(f"  {out.name}: {dims}")

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ Verified ONNX model: {output_path} ({size_mb:.2f} MB)")
    print("\nExpected by mediapipeDetector.ts:")
    print("  Input  [0]: [1, 128, 128, 3]  → NHWC, normalized to [-1, 1]")
    print("  Output [0]: [1, 896, 16]       → raw box regressors")
    print("  Output [1]: [1, 896,  1]       → raw logit scores")
    print("\nIf the output shapes differ, update N_ANCHORS / output parsing")
    print("in mediapipeDetector.ts to match the actual model outputs.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BlazeFace TFLite → ONNX")
    parser.add_argument(
        "--output",
        default=OUTPUT_PATH_DEFAULT,
        help="Destination path for the ONNX file",
    )
    args = parser.parse_args()

    tflite_path = find_or_download_tflite()
    convert_tflite_to_onnx(tflite_path, args.output)
    simplify_onnx(args.output)
    verify_onnx(args.output)
