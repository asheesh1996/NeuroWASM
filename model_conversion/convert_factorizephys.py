"""
Convert FactorizePhys model from rPPG-Toolbox to ONNX format.

FactorizePhys uses MD-ST (multi-dimensional spatiotemporal) factorized attention.
Its internal forward() does torch.diff(x, dim=2) reducing temporal length by 1,
so BVP_LENGTH = INPUT_FRAMES - 1.

Model parameters (from rPPG-Toolbox architecture):
  - img_size    = 72  (hard-coded by BVP_Head convolution chain: must be 72×72)
  - T           = 160 (INPUT_FRAMES)
  - BVP_LENGTH  = 159 (= T - 1, due to internal torch.diff)

ONNX wrapper:
  Input:  [1, INPUT_FRAMES, 3, 72, 72]   (time-first layout from TypeScript)
  Output: [1, BVP_LENGTH]                (= [1, 159])

Checkpoint download:
  Pre-trained UBFC-rPPG checkpoint is available from the rPPG-Toolbox release
  or from the authors' Google Drive.  Set --checkpoint to its local path, or
  let gdown fetch it automatically if CHECKPOINT_GDRIVE_ID is set below.
  Alternatively the model exports fine with random weights for architecture
  verification; omit the checkpoint to do a dry run.

Usage:
  python convert_factorizephys.py [--checkpoint /path/to/weights.pth]
"""

import sys
import os
import argparse
import types
import torch
import torch.nn as nn
import onnx
from onnx import checker

TOOLBOX_DIR = "/workspace/rPPG-Toolbox"
sys.path.insert(0, TOOLBOX_DIR)

# Model parameters
IMG_SIZE = 72             # Must match BVP_Head conv architecture (72×72 only)
INPUT_FRAMES = 160
BVP_LENGTH = INPUT_FRAMES - 1   # 159 — FactorizePhys does internal torch.diff

# Optional: set this to a gdown file ID to auto-download the checkpoint.
CHECKPOINT_GDRIVE_ID = ""


class OnnxFriendlyTSM(nn.Module):
    """
    ONNX-friendly Temporal Shift Module.

    Replaces the standard TSM that calls torch.zeros_like() on a full-size
    tensor (which bakes a giant constant into the ONNX graph).  Uses
    slice+concat with scalar-zero multiplication instead.
    """

    def __init__(self, n_segment=10, fold_div=3):
        super().__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div

        left = x[:, :, :fold]
        left_shifted = torch.cat([left[:, 1:], left[:, -1:] * 0], dim=1)

        right = x[:, :, fold:2 * fold]
        right_shifted = torch.cat([right[:, :1] * 0, right[:, :-1]], dim=1)

        no_shift = x[:, :, 2 * fold:]

        out = torch.cat([left_shifted, right_shifted, no_shift], dim=2)
        return out.view(nt, c, h, w)


def replace_tsm_modules(module: nn.Module, frame_depth: int) -> None:
    """
    Recursively replace all TSM instances in the module with OnnxFriendlyTSM.
    Works regardless of the attribute name used by the model.
    """
    for name, child in list(module.named_children()):
        cls_name = type(child).__name__
        if "TSM" in cls_name or "TemporalShift" in cls_name:
            # Infer n_segment and fold_div from the original module if available
            n_seg = getattr(child, "n_segment", frame_depth)
            fold  = getattr(child, "fold_div", 3)
            setattr(module, name, OnnxFriendlyTSM(n_segment=n_seg, fold_div=fold))
            print(f"  Replaced {cls_name} → OnnxFriendlyTSM (n_segment={n_seg}, fold_div={fold})")
        else:
            replace_tsm_modules(child, frame_depth)


def _patch_forward_no_diff(fp_model: nn.Module) -> None:
    """
    Replace FactorizePhys.forward with a version that expects already-diffed input.
    The wrapper performs torch.diff via ONNX-compatible slice subtraction before
    calling the model, so we need to skip the internal torch.diff call.
    """
    def forward_no_diff(self, x):
        # x: [B, C, T-1, H, W] — already differenced by OnnxWrapper
        batch, _, length, _, _ = x.shape

        if self.in_channels == 3:
            x = self.norm(x[:, :3, :, :, :])
        elif self.in_channels == 1:
            x = self.norm(x[:, -1:, :, :, :])
        elif self.in_channels == 4:
            rgb_x = self.rgb_norm(x[:, :3, :, :, :])
            thermal_x = self.thermal_norm(x[:, -1:, :, :, :])
            x = torch.concat([rgb_x, thermal_x], dim=1)

        voxel_embeddings = self.rppg_feature_extractor(x)
        # eval mode, md_infer=False, debug=False → simple branch (no FSAM return)
        bvp_signal = self.rppg_head(voxel_embeddings, batch, length)
        return bvp_signal, voxel_embeddings

    fp_model.forward = types.MethodType(forward_no_diff, fp_model)


class OnnxWrapper(nn.Module):
    """
    Minimal wrapper that:
      - Accepts time-first input [1, T, C, H, W] (layout used by TypeScript)
      - Permutes to channels-first [1, C, T, H, W] for FactorizePhys
      - Applies temporal diff via slice subtraction (ONNX-compatible, replaces aten::diff)
      - Extracts only rPPG from the (rPPG, voxel_embeddings) tuple FactorizePhys returns
      - Returns [1, T-1] BVP signal
    """

    def __init__(self, fp_model: nn.Module):
        super().__init__()
        self.fp_model = fp_model
        # Patch model to skip internal torch.diff (not ONNX-exportable)
        _patch_forward_no_diff(fp_model)

    def forward(self, x):
        # x: [1, T, C, H, W] — time-first layout from TypeScript
        x = x.permute(0, 2, 1, 3, 4)                    # → [1, C, T, H, W]
        x = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]     # ONNX-safe diff → [1, C, T-1, H, W]
        outputs = self.fp_model(x)                        # returns (rPPG, voxel_embeddings)
        bvp_signal = outputs[0]                            # [1, T-1]
        return bvp_signal


def build_model():
    """Load FactorizePhys architecture from rPPG-Toolbox with correct constructor signature."""
    from neural_methods.model.FactorizePhys.FactorizePhys import FactorizePhys, model_config as default_config  # noqa: PLC0415

    # Build md_config from defaults, overriding spatial/temporal dims for our use
    md_config = dict(default_config)
    md_config.update({
        'MD_FSAM': True,
        'MD_INFERENCE': False,
        'height': IMG_SIZE,
        'weight': IMG_SIZE,
        'frames': INPUT_FRAMES,
    })

    fp_model = FactorizePhys(frames=INPUT_FRAMES, md_config=md_config)
    print(f"✅ Loaded FactorizePhys (frames={INPUT_FRAMES}, img_size={IMG_SIZE})")
    return fp_model


def load_weights(fp_model: nn.Module, checkpoint_path: str | None) -> bool:
    """Load pre-trained weights if a path is supplied."""
    if checkpoint_path:
        paths = [checkpoint_path]
    else:
        paths = [
            os.path.join(TOOLBOX_DIR, "final_model_release", "UBFC-rPPG_FactorizePhys.pth"),
            os.path.join(TOOLBOX_DIR, "checkpoints", "FactorizePhys_UBFC.pth"),
            "/workspace/pretrained/FactorizePhys_UBFC.pth",
        ]

    for path in paths:
        if path and os.path.exists(path):
            print(f"Loading weights from: {path}")
            state = torch.load(path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            fp_model.load_state_dict(state, strict=False)
            print("✅ Pretrained weights loaded")
            return True

    # Optional: download from Google Drive
    if CHECKPOINT_GDRIVE_ID:
        try:
            import gdown  # noqa: PLC0415
            gdrive_path = "/workspace/pretrained/FactorizePhys_UBFC.pth"
            os.makedirs(os.path.dirname(gdrive_path), exist_ok=True)
            gdown.download(id=CHECKPOINT_GDRIVE_ID, output=gdrive_path, quiet=False)
            state = torch.load(gdrive_path, map_location="cpu")
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            fp_model.load_state_dict(state, strict=False)
            print("✅ Pretrained weights downloaded and loaded")
            return True
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  gdown download failed: {e}")

    print("⚠️  No pretrained weights found — using random weights (architecture only)")
    return False


def export_onnx(fp_model: nn.Module) -> str:
    fp_model.eval()

    # FactorizePhys uses FSAM (not TSM), no TSM replacement needed.
    wrapper = OnnxWrapper(fp_model)
    wrapper.eval()

    output_path = "/workspace/factorizephys.onnx"
    dummy = torch.randn(1, INPUT_FRAMES, 3, IMG_SIZE, IMG_SIZE)

    print(f"\nTest forward pass with dummy input {list(dummy.shape)} ...")
    with torch.no_grad():
        out = wrapper(dummy)
        print(f"  Input:  {list(dummy.shape)}")
        print(f"  Output: {list(out.shape)}")

    assert list(out.shape) == [1, BVP_LENGTH], (
        f"Unexpected output shape {list(out.shape)}, expected [1, {BVP_LENGTH}]"
    )

    print("\nExporting ONNX...")
    torch.onnx.export(
        wrapper,
        dummy,
        output_path,
        opset_version=17,
        input_names=["video_frames"],
        output_names=["bvp_signal"],
        dynamic_axes=None,
    )

    onnx_model = onnx.load(output_path)
    checker.check_model(onnx_model)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"\n✅ ONNX exported: {output_path} ({size_mb:.1f} MB)")
    print(f"   Input:  video_frames  [1, {INPUT_FRAMES}, 3, {IMG_SIZE}, {IMG_SIZE}]")
    print(f"   Output: bvp_signal    [1, {BVP_LENGTH}]")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert FactorizePhys to ONNX")
    parser.add_argument("--checkpoint", default=None, help="Path to pretrained .pth checkpoint")
    args = parser.parse_args()

    fp_model = build_model()
    load_weights(fp_model, args.checkpoint)
    export_onnx(fp_model)
