# Model Conversion for NeuroWASM rPPG

Converts **EfficientPhys** and **FactorizePhys** (rPPG-Toolbox) to ONNX,
downloads **UltraFace-320**, and converts **BlazeFace** (MediaPipe) for face detection.

## Quick Start (Docker)

```bash
cd model_conversion

# Build the converter image
docker build -t neurowasm-converter .

# Run all conversions — outputs land in ../public/
docker run --rm -v "$(pwd)/../public:/output" neurowasm-converter
```

This produces four files in `public/`:

| File | Model | Shape (input → output) |
|------|-------|------------------------|
| `factorizephys.onnx` | FactorizePhys (default rPPG) | `[1,160,3,128,128]` → `[1,160]` |
| `efficientphys.onnx` | EfficientPhys (legacy rPPG) | `[1,151,3,72,72]` → `[1,150]` |
| `blazeface_128.onnx` | BlazeFace short-range (default face) | `[1,128,128,3]` NHWC → boxes+scores |
| `ultraface_320.onnx` | UltraFace-320 (legacy face) | `[1,3,240,320]` → boxes+scores |

## Pretrained Weights

The conversion tries to find pretrained weights in the rPPG-Toolbox clone.
If none are found, it exports with random weights (good for pipeline testing only).

### FactorizePhys

Download from the [rPPG-Toolbox releases](https://github.com/ubicomplab/rPPG-Toolbox/releases)
and place in `model_conversion/pretrained/FactorizePhys_UBFC.pth`, then:

```bash
docker run --rm \
  -v "$(pwd)/pretrained:/workspace/pretrained" \
  -v "$(pwd)/../public:/output" \
  neurowasm-converter
```

Or provide the checkpoint explicitly:

```bash
docker run --rm -it \
  -v "$(pwd)/pretrained:/workspace/pretrained" \
  -v "$(pwd)/../public:/output" \
  neurowasm-converter bash
# Inside the container:
python convert_factorizephys.py --checkpoint /workspace/pretrained/MyCheckpoint.pth
```

### EfficientPhys

Same pattern; place weights at `model_conversion/pretrained/EfficientPhys_UBFC.pth`.

## Model Details

### FactorizePhys (rPPG — default)

- **Input**: `[1, 160, 3, 128, 128]` — 160 face crops (128×128, RGB, z-score normalized)
- **Output**: `[1, 160]` — BVP signal (no internal frame-difference)
- **Architecture**: MD-ST factorized attention + Temporal Shift Module
- **Warmup**: ~160 frames ≈ 5.3 s at 30 fps

### EfficientPhys (rPPG — legacy)

- **Input**: `[1, 151, 3, 72, 72]` — 151 face crops (72×72, RGB, z-score normalized)
- **Output**: `[1, 150]` — BVP signal (internal `torch.diff` reduces by 1)
- **Architecture**: Dual-stream TSM with attention gating
- **Warmup**: ~151 frames ≈ 5 s at 30 fps

### BlazeFace short-range (face detector — default)

- **Input**: `[1, 128, 128, 3]` — NHWC, normalized via `(pixel − 127.5) / 127.5`
- **Outputs**: `[1, 896, 16]` regressors + `[1, 896, 1]` logit scores
- **Anchor decoding**: performed in TypeScript (`mediapipeDetector.ts`)
- **Anchor config**: strides `[8,16,16,16]`, 2 anchors per cell → 896 total

### UltraFace-320 (face detector — legacy)

- **Input**: `[1, 3, 240, 320]` — NCHW, normalized `(pixel − 127) / 128`
- **Output**: `confidences [1, N, 2]` + `boxes [1, N, 4]`
- **Size**: ~1.2 MB (ONNX)
