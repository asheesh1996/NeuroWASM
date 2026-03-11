// mediapipeDetector.ts
// MediaPipe BlazeFace short-range face detector (ONNX backend).
//
// Expected ONNX model: blazeface_128.onnx, produced by convert_mediapipe.py.
// Model input:  [1, 3, 128, 128] float32, values in [-1, 1] (NCHW layout)
//               Normalization: (pixel - 127.5) / 127.5
// Model outputs (tflite2onnx order):
//   outputNames[0] — regressors   [1, 896, 16]  raw box offsets (y,x,h,w + keypoints)
//   outputNames[1] — classificators [1, 896, 1] raw logit scores (pre-sigmoid)
//
// Anchor config: 896 anchors, strides [8, 16, 16, 16], 2 anchors per grid cell.
// Box decoding (from MediaPipe tensors_to_detections_calculator, fixed_anchor_size=true):
//   y_center = raw[0] / 128 + anchor_cy
//   x_center = raw[1] / 128 + anchor_cx
//   h        = exp(raw[2] / 128)
//   w        = exp(raw[3] / 128)
//   (all coordinates normalized to [0, 1])

import * as ort from 'onnxruntime-web/webgpu';
import type { FaceResult, IFaceDetector } from './faceDetector';

// Model constants
const INPUT_WH = 128;
const N_ANCHORS = 896;
const SCORE_THRESHOLD = 0.3; // applied after sigmoid (logit ≈ −0.85)

/**
 * Generate BlazeFace short-range anchors (896 entries).
 * Returns Float32Array of length N_ANCHORS * 2, packed as [cx0,cy0, cx1,cy1, ...].
 */
function generateBlazeFaceAnchors(): Float32Array {
  // strides [8, 16, 16, 16] × 2 anchors each:
  //   stride  8: 16×16 grid × 2 = 512
  //   stride 16: 8×8 grid × 2 × 3 layers = 384
  //   total: 896
  const strides = [8, 16, 16, 16];
  const anchors = new Float32Array(N_ANCHORS * 2);
  let idx = 0;
  for (const stride of strides) {
    const gridSize = Math.ceil(INPUT_WH / stride);
    for (let y = 0; y < gridSize; y++) {
      for (let x = 0; x < gridSize; x++) {
        for (let a = 0; a < 2; a++) {
          anchors[idx++] = (x + 0.5) / gridSize; // cx (normalized)
          anchors[idx++] = (y + 0.5) / gridSize; // cy (normalized)
        }
      }
    }
  }
  return anchors;
}

const ANCHORS = generateBlazeFaceAnchors();

export class MediaPipeDetector implements IFaceDetector {
  private session: ort.InferenceSession | null = null;
  private prevCenter: { x: number; y: number } | null = null;
  private readonly stabilityThreshold = 12; // px movement considered "unstable"

  // Offscreen canvas — 128×128
  private readonly preprocessCanvas: OffscreenCanvas;
  private readonly preprocessCtx: OffscreenCanvasRenderingContext2D;

  // Pre-allocated pixel buffer — NCHW [1, 3, 128, 128] = 49,152 floats
  private readonly preprocessBuffer: Float32Array;

  constructor() {
    this.preprocessCanvas = new OffscreenCanvas(INPUT_WH, INPUT_WH);
    this.preprocessCtx = this.preprocessCanvas.getContext('2d', { willReadFrequently: true })!;
    this.preprocessBuffer = new Float32Array(INPUT_WH * INPUT_WH * 3);
  }

  async init(backend: 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm' = 'webgpu-high-performance'): Promise<boolean> {
    try {
      const modelUrl = `${import.meta.env.BASE_URL}blazeface_128.onnx`;

      if (backend === 'wasm') {
        this.session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
        });
        console.log('%c✅ Face detector loaded (BlazeFace-128) [WASM]', 'color: #ffcc00; font-weight: bold');
      } else {
        const powerPreference = backend === 'webgpu-high-performance' ? 'high-performance' : 'low-power';
        try {
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: [{ name: 'webgpu', powerPreference } as any],
          });
          console.log('%c✅ Face detector loaded (BlazeFace-128) [WebGPU]', 'color: #00ff88; font-weight: bold');
        } catch (e) {
          console.warn('[MediaPipeDetector] WebGPU failed, falling back to WASM:', e);
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
          });
          console.log('%c✅ Face detector loaded (BlazeFace-128) [WASM fallback]', 'color: #ffcc00; font-weight: bold');
        }
      }

      console.log('📥 MediaPipeDetector inputs:', this.session.inputNames);
      console.log('📤 MediaPipeDetector outputs:', this.session.outputNames);
      return true;
    } catch (error) {
      console.error('Failed to load BlazeFace model:', error);
      return false;
    }
  }

  async detect(video: HTMLVideoElement): Promise<FaceResult | null> {
    if (!this.session) return null;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (vw === 0 || vh === 0) return null;

    // Preprocess: resize video to 128×128
    this.preprocessCtx.drawImage(video, 0, 0, INPUT_WH, INPUT_WH);
    const imageData = this.preprocessCtx.getImageData(0, 0, INPUT_WH, INPUT_WH);

    // Convert to NCHW float32, normalize to [-1, 1] via (pixel - 127.5) / 127.5
    const pixelCount = INPUT_WH * INPUT_WH;
    const buf = this.preprocessBuffer;
    for (let i = 0; i < pixelCount; i++) {
      buf[i]                  = (imageData.data[i * 4]     - 127.5) / 127.5; // R channel
      buf[pixelCount + i]     = (imageData.data[i * 4 + 1] - 127.5) / 127.5; // G channel
      buf[2 * pixelCount + i] = (imageData.data[i * 4 + 2] - 127.5) / 127.5; // B channel
    }

    const inputTensor = new ort.Tensor('float32', buf, [1, 3, INPUT_WH, INPUT_WH]);
    const feeds: Record<string, ort.Tensor> = {};
    feeds[this.session.inputNames[0]] = inputTensor;

    const outputs = await this.session.run(feeds);

    // Identify outputs by shape: regressors [.., 16] and scores [.., 1]
    const out0 = outputs[this.session.outputNames[0]];
    const out1 = outputs[this.session.outputNames[1]];

    let regressors: Float32Array;
    let scores: Float32Array;

    if (out0.dims[out0.dims.length - 1] === 16) {
      regressors = out0.data as Float32Array;
      scores     = out1.data as Float32Array;
    } else {
      regressors = out1.data as Float32Array;
      scores     = out0.data as Float32Array;
    }

    // Find highest-confidence detection above threshold
    let bestScore = 0;
    let bestIdx = -1;
    let rawMaxLogit = -Infinity;

    for (let i = 0; i < N_ANCHORS; i++) {
      const logit = scores[i];
      if (logit > rawMaxLogit) rawMaxLogit = logit;
      const prob = 1 / (1 + Math.exp(-logit)); // sigmoid
      if (prob > bestScore && prob > SCORE_THRESHOLD) {
        bestScore = prob;
        bestIdx = i;
      }
    }
    console.debug(`[BlazeFace] maxLogit=${rawMaxLogit.toFixed(3)} bestIdx=${bestIdx} bestScore=${bestScore.toFixed(3)}`);

    if (bestIdx < 0) {
      this.prevCenter = null;
      return null;
    }

    // Decode box using MediaPipe anchor convention (fixed_anchor_size=true):
    //   raw[0] = y_offset, raw[1] = x_offset, raw[2] = raw_h, raw[3] = raw_w
    //   All values are linear (divide by INPUT_WH), no exp() — per tensors_to_detections_calculator
    const anchorCx = ANCHORS[bestIdx * 2];
    const anchorCy = ANCHORS[bestIdx * 2 + 1];
    const base = bestIdx * 16;

    const yCtr = regressors[base + 0] / INPUT_WH + anchorCy;
    const xCtr = regressors[base + 1] / INPUT_WH + anchorCx;
    const h    = regressors[base + 2] / INPUT_WH;
    const w    = regressors[base + 3] / INPUT_WH;

    // Convert center-form to corner-form, clamp to [0,1]
    const x1n = Math.max(0, xCtr - w / 2);
    const y1n = Math.max(0, yCtr - h / 2);
    const x2n = Math.min(1, xCtr + w / 2);
    const y2n = Math.min(1, yCtr + h / 2);

    const bbox = {
      x: x1n * vw,
      y: y1n * vh,
      w: (x2n - x1n) * vw,
      h: (y2n - y1n) * vh,
    };

    // Stability check
    const cx = bbox.x + bbox.w / 2;
    const cy = bbox.y + bbox.h / 2;
    let stable = true;

    if (this.prevCenter) {
      const dist = Math.hypot(cx - this.prevCenter.x, cy - this.prevCenter.y);
      stable = dist < this.stabilityThreshold;
    }
    this.prevCenter = { x: cx, y: cy };

    return { bbox, confidence: bestScore, stable };
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.prevCenter = null;
  }
}
