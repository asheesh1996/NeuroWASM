// factorizePhysProcessor.ts
// Manages the frame buffer and FactorizePhys ONNX inference for rPPG signal extraction.
//
// Expected ONNX model: factorizephys.onnx, produced by convert_factorizephys.py.
// Model input:  [1, 160, 3, 72, 72] float32, z-score standardized (NCHW per frame)
// Model output: [1, 159] BVP signal
//
// FactorizePhys expects [B, C, T, H, W] input (channels-first temporal); the ONNX wrapper
// handles the permutation from time-first [B, T, C, H, W] used here.
// Internal torch.diff reduces temporal length by 1: INPUT_FRAMES=160 → BVP_LENGTH=159.

import * as ort from 'onnxruntime-web/webgpu';
import type { FaceResult } from './faceDetector';
import type { IRppgProcessor } from './rppgProcessor';

/** Number of input frames FactorizePhys expects */
export const FP_INPUT_FRAMES = 160;
/** BVP output length (INPUT_FRAMES - 1 due to internal torch.diff in FactorizePhys) */
export const FP_BVP_LENGTH = 159;
/** Face crop size for FactorizePhys (must be 72 — hard-coded by BVP_Head conv chain) */
export const FP_FACE_SIZE = 72;
/** New frames required before the next sliding-window inference */
export const FP_SLIDE_STEP = 30;

export class FactorizePhysProcessor implements IRppgProcessor {
  private session: ort.InferenceSession | null = null;
  private framesSinceLastInference = 0;
  private isProcessing = false;

  // Pre-allocated ring buffer — FP_INPUT_FRAMES slots of [3, 72, 72] each.
  private readonly frameRing: Float32Array[];
  private readonly tsRing: Float64Array;
  private ringHead = 0;
  private ringCount = 0;

  // Pre-allocated inference input — avoids ~31 MB alloc per inference.
  private readonly inputBuffer: Float32Array;

  // Offscreen canvas for face cropping
  private readonly cropCanvas: OffscreenCanvas;
  private readonly cropCtx: OffscreenCanvasRenderingContext2D;

  constructor() {
    this.cropCanvas = new OffscreenCanvas(FP_FACE_SIZE, FP_FACE_SIZE);
    this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true })!;

    const pixelsPerFrame = 3 * FP_FACE_SIZE * FP_FACE_SIZE; // 15,552
    this.frameRing = Array.from({ length: FP_INPUT_FRAMES }, () => new Float32Array(pixelsPerFrame));
    this.tsRing = new Float64Array(FP_INPUT_FRAMES);
    this.inputBuffer = new Float32Array(FP_INPUT_FRAMES * pixelsPerFrame); // ~9.7 MB
  }

  async init(backend: 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm' = 'webgpu-high-performance'): Promise<boolean> {
    try {
      const modelUrl = `${import.meta.env.BASE_URL}factorizephys.onnx`;

      if (backend === 'wasm') {
        console.log('[FactorizePhysProcessor] Loading model with EP: wasm');
        this.session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
        });
        console.log('%c✅ rPPG model loaded (FactorizePhys) [WASM]', 'color: #ffcc00; font-weight: bold');
      } else {
        const powerPreference = backend === 'webgpu-high-performance' ? 'high-performance' : 'low-power';
        try {
          console.log(`[FactorizePhysProcessor] Loading model with EP: webgpu (${powerPreference})`);
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: [{ name: 'webgpu', powerPreference } as any],
          });
          console.log('%c✅ rPPG model loaded (FactorizePhys) [WebGPU]', 'color: #00ff88; font-weight: bold');
        } catch (e) {
          console.warn('[FactorizePhysProcessor] WebGPU failed, falling back to WASM:', e);
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
          });
          console.log('%c✅ rPPG model loaded (FactorizePhys) [WASM fallback]', 'color: #ffcc00; font-weight: bold');
        }
      }

      console.log('📥 FactorizePhysProcessor inputs:', this.session.inputNames);
      console.log('📤 FactorizePhysProcessor outputs:', this.session.outputNames);
      return true;
    } catch (error) {
      console.error('Failed to load FactorizePhys model:', error);
      return false;
    }
  }

  /**
   * Add a video frame to the buffer. Extracts and preprocesses the face ROI.
   * Raw [0,255] values are stored; z-score is applied at inference time.
   */
  addFrame(video: HTMLVideoElement, face: FaceResult): ImageData {
    const { bbox } = face;

    const margin = 0.2;
    const mx = bbox.w * margin;
    const my = bbox.h * margin;
    const cropX = Math.max(0, bbox.x - mx);
    const cropY = Math.max(0, bbox.y - my);
    const cropW = Math.min(video.videoWidth - cropX, bbox.w + 2 * mx);
    const cropH = Math.min(video.videoHeight - cropY, bbox.h + 2 * my);

    this.cropCtx.drawImage(
      video,
      cropX, cropY, cropW, cropH,
      0, 0, FP_FACE_SIZE, FP_FACE_SIZE,
    );

    const imageData = this.cropCtx.getImageData(0, 0, FP_FACE_SIZE, FP_FACE_SIZE);

    const pixelCount = FP_FACE_SIZE * FP_FACE_SIZE;
    const slot = this.frameRing[this.ringHead];
    for (let i = 0; i < pixelCount; i++) {
      slot[i]                  = imageData.data[i * 4];     // R
      slot[pixelCount + i]     = imageData.data[i * 4 + 1]; // G
      slot[2 * pixelCount + i] = imageData.data[i * 4 + 2]; // B
    }
    this.tsRing[this.ringHead] = performance.now();
    this.ringHead = (this.ringHead + 1) % FP_INPUT_FRAMES;
    if (this.ringCount < FP_INPUT_FRAMES) this.ringCount++;
    this.framesSinceLastInference++;

    return imageData;
  }

  get isReady(): boolean {
    return this.ringCount >= FP_INPUT_FRAMES;
  }

  get shouldInfer(): boolean {
    return this.isReady && this.framesSinceLastInference >= FP_SLIDE_STEP && !this.isProcessing;
  }

  get bufferedFrames(): number {
    return this.ringCount;
  }

  get warmupFrames(): number {
    return FP_INPUT_FRAMES;
  }

  getSampleRate(): number {
    if (this.ringCount < 2) return 30;
    const frames = Math.min(this.ringCount, FP_INPUT_FRAMES);
    const startSlot = (this.ringHead - frames + FP_INPUT_FRAMES) % FP_INPUT_FRAMES;
    const endSlot   = (this.ringHead - 1 + FP_INPUT_FRAMES) % FP_INPUT_FRAMES;
    const elapsed = (this.tsRing[endSlot] - this.tsRing[startSlot]) / 1000;
    return elapsed > 0 ? (frames - 1) / elapsed : 30;
  }

  /**
   * Run FactorizePhys inference on the buffered frames.
   *
   * Input tensor:  [1, FP_INPUT_FRAMES, 3, 72, 72] — z-score standardized
   * Output tensor: [1, FP_BVP_LENGTH]
   */
  async runInference(): Promise<Float32Array | null> {
    if (!this.session || !this.isReady || this.isProcessing) return null;
    this.isProcessing = true;

    try {
      const pixelsPerFrame = 3 * FP_FACE_SIZE * FP_FACE_SIZE;
      const totalSize = FP_INPUT_FRAMES * pixelsPerFrame;
      const inputData = this.inputBuffer;

      // Assemble contiguous input from ring buffer in chronological order
      for (let t = 0; t < FP_INPUT_FRAMES; t++) {
        const slot = (this.ringHead + t) % FP_INPUT_FRAMES;
        inputData.set(this.frameRing[slot], t * pixelsPerFrame);
      }

      // Z-score standardization — Welford single-pass
      let mean = 0, M2 = 0;
      for (let i = 0; i < totalSize; i++) {
        const delta = inputData[i] - mean;
        mean += delta / (i + 1);
        M2 += delta * (inputData[i] - mean);
      }
      const std = Math.sqrt(M2 / totalSize) || 1;

      for (let i = 0; i < totalSize; i++) {
        inputData[i] = (inputData[i] - mean) / std;
      }

      const inputTensor = new ort.Tensor(
        'float32',
        inputData,
        [1, FP_INPUT_FRAMES, 3, FP_FACE_SIZE, FP_FACE_SIZE],
      );

      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.session.inputNames[0]] = inputTensor;

      const outputs = await this.session.run(feeds);
      const bvpTensor = outputs[this.session.outputNames[0]];
      const bvpSignal = bvpTensor.data as Float32Array;

      this.framesSinceLastInference = 0;
      return bvpSignal;
    } catch (error) {
      console.error('FactorizePhys inference error:', error);
      return null;
    } finally {
      this.isProcessing = false;
    }
  }

  reset(): void {
    this.ringHead = 0;
    this.ringCount = 0;
    this.framesSinceLastInference = 0;
    this.isProcessing = false;
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.reset();
  }
}
