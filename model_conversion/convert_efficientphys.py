"""
Convert EfficientPhys model from rPPG-Toolbox to ONNX format.

The UBFC-rPPG pretrained model uses:
  - frame_depth = 10 (TSM segment size, from training config)
  - img_size    = 72

The toolbox model's forward() expects 4D input [T, C, H, W]:
  1. torch.diff(inputs, dim=0) → [T-1, C, H, W]
  2. BatchNorm2d → TSM → convolutions → attention → dense → [T-1, 1]

The trainer feeds N*D frames collapsed to 4D, truncated to a multiple of
frame_depth, plus one extra frame so that after diff the count is still
a multiple of frame_depth.

For ONNX we wrap the model with a batch dimension:
  Input:  [1, INPUT_FRAMES, 3, 72, 72]  (INPUT_FRAMES = BVP_LENGTH + 1 = 151)
  Output: [1, BVP_LENGTH]               (BVP_LENGTH = 150, divisible by 10)
"""

import sys
import os
import torch
import torch.nn as nn
import onnx
from onnx import checker

TOOLBOX_DIR = "/workspace/rPPG-Toolbox"
sys.path.insert(0, TOOLBOX_DIR)

# Model parameters (from UBFC-rPPG training config)
FRAME_DEPTH = 10
IMG_SIZE = 72
BVP_LENGTH = 150                    # output signal length (must be multiple of FRAME_DEPTH)
INPUT_FRAMES = BVP_LENGTH + 1       # one extra frame consumed by torch.diff


class OnnxFriendlyTSM(nn.Module):
    """
    ONNX-friendly TSM that avoids torch.zeros_like() on full-size tensors.

    The original TSM does:
      out = torch.zeros_like(x)           ← creates 95+ MB constant when traced
      out[:, :-1, :fold] = x[:, 1:, :fold]
      out[:, 1:, fold:2*fold] = x[:, :-1, fold:2*fold]
      out[:, :, 2*fold:] = x[:, :, 2*fold:]

    This version uses slice+concat with *scalar* multiplication by zero,
    producing tiny constants instead of full-tensor zeros.
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

        # Shift left: x[:, 1:, :fold] fills positions :-1, last position is zero
        left = x[:, :, :fold]
        left_shifted = torch.cat([left[:, 1:], left[:, -1:] * 0], dim=1)

        # Shift right: x[:, :-1, fold:2*fold] fills positions 1:, first position is zero
        right = x[:, :, fold:2 * fold]
        right_shifted = torch.cat([right[:, :1] * 0, right[:, :-1]], dim=1)

        # No shift for remaining channels
        no_shift = x[:, :, 2 * fold:]

        out = torch.cat([left_shifted, right_shifted, no_shift], dim=2)
        return out.view(nt, c, h, w)


class OnnxWrapper(nn.Module):
    """
    Wraps the toolbox model for ONNX export.

    - Adds a batch dimension: input [1, T+1, C, H, W] → output [1, T]
    - Replaces torch.diff (not supported in ONNX) with equivalent slicing
    - Replaces TSM modules with ONNX-friendly versions (no torch.zeros_like)
    """

    def __init__(self, model):
        super().__init__()
        # Copy all sub-modules so weights are exported
        self.batch_norm = model.batch_norm
        # Replace TSM modules with ONNX-friendly versions
        self.TSM_1 = OnnxFriendlyTSM(n_segment=model.TSM_1.n_segment, fold_div=model.TSM_1.fold_div)
        self.TSM_2 = OnnxFriendlyTSM(n_segment=model.TSM_2.n_segment, fold_div=model.TSM_2.fold_div)
        self.TSM_3 = OnnxFriendlyTSM(n_segment=model.TSM_3.n_segment, fold_div=model.TSM_3.fold_div)
        self.TSM_4 = OnnxFriendlyTSM(n_segment=model.TSM_4.n_segment, fold_div=model.TSM_4.fold_div)
        self.motion_conv1 = model.motion_conv1
        self.motion_conv2 = model.motion_conv2
        self.motion_conv3 = model.motion_conv3
        self.motion_conv4 = model.motion_conv4
        self.apperance_att_conv1 = model.apperance_att_conv1
        self.attn_mask_1 = model.attn_mask_1
        self.apperance_att_conv2 = model.apperance_att_conv2
        self.attn_mask_2 = model.attn_mask_2
        self.avg_pooling_1 = model.avg_pooling_1
        self.avg_pooling_3 = model.avg_pooling_3
        self.dropout_1 = model.dropout_1
        self.dropout_3 = model.dropout_3
        self.dropout_4 = model.dropout_4
        self.final_dense_1 = model.final_dense_1
        self.final_dense_2 = model.final_dense_2

    def forward(self, x):
        # x: [1, T+1, C, H, W]
        x = x.squeeze(0)               # [T+1, C, H, W]

        # Replace torch.diff with ONNX-compatible slicing
        x = x[1:] - x[:-1]             # [T, C, H, W]

        x = self.batch_norm(x)

        network_input = self.TSM_1(x)
        d1 = torch.tanh(self.motion_conv1(network_input))
        d1 = self.TSM_2(d1)
        d2 = torch.tanh(self.motion_conv2(d1))

        g1 = torch.sigmoid(self.apperance_att_conv1(d2))
        g1 = self.attn_mask_1(g1)
        gated1 = d2 * g1

        d3 = self.avg_pooling_1(gated1)
        d4 = self.dropout_1(d3)

        d4 = self.TSM_3(d4)
        d5 = torch.tanh(self.motion_conv3(d4))
        d5 = self.TSM_4(d5)
        d6 = torch.tanh(self.motion_conv4(d5))

        g2 = torch.sigmoid(self.apperance_att_conv2(d6))
        g2 = self.attn_mask_2(g2)
        gated2 = d6 * g2

        d7 = self.avg_pooling_3(gated2)
        d8 = self.dropout_3(d7)
        d9 = d8.view(d8.size(0), -1)
        d10 = torch.tanh(self.final_dense_1(d9))
        d11 = self.dropout_4(d10)
        out = self.final_dense_2(d11)   # [T, 1]

        return out.squeeze(-1).unsqueeze(0)  # [1, T]


def build_model():
    from neural_methods.model.EfficientPhys import EfficientPhys

    model = EfficientPhys(frame_depth=FRAME_DEPTH, img_size=IMG_SIZE)
    print(f"✅ Loaded EfficientPhys (frame_depth={FRAME_DEPTH}, img_size={IMG_SIZE})")
    return model


def load_weights(model):
    possible_paths = [
        os.path.join(TOOLBOX_DIR, "final_model_release", "UBFC-rPPG_EfficientPhys.pth"),
        os.path.join(TOOLBOX_DIR, "checkpoints", "EfficientPhys_UBFC.pth"),
        "/workspace/pretrained/EfficientPhys_UBFC.pth",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading weights from: {path}")
            state = torch.load(path, map_location="cpu")
            if "state_dict" in state:
                state = state["state_dict"]
            model.load_state_dict(state, strict=False)
            print("✅ Pretrained weights loaded")
            return True

    print("⚠️  No pretrained weights found — using random weights")
    return False


def export_onnx(model):
    model.eval()
    wrapper = OnnxWrapper(model)
    wrapper.eval()

    output_path = "/workspace/efficientphys.onnx"
    dummy = torch.randn(1, INPUT_FRAMES, 3, IMG_SIZE, IMG_SIZE)

    print(f"\nTest forward pass...")
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
    print(f"   Input:  video_frames [1, {INPUT_FRAMES}, 3, {IMG_SIZE}, {IMG_SIZE}]")
    print(f"   Output: bvp_signal   [1, {BVP_LENGTH}]")
    return output_path


if __name__ == "__main__":
    model = build_model()
    load_weights(model)
    export_onnx(model)
