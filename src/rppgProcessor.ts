// rppgProcessor.ts
// Manages the frame buffer and EfficientPhys ONNX inference for rPPG signal extraction.

import * as ort from 'onnxruntime-web/webgpu';
import type { FaceResult } from './faceDetector';

/**
 * Number of input frames the ONNX model expects.
 * The model internally computes torch.diff (frame differences),
 * so INPUT_FRAMES − 1 = BVP_LENGTH must be a multiple of FRAME_DEPTH (10).
 */
export const INPUT_FRAMES = 151;
/** Number of BVP signal samples the model outputs */
export const BVP_LENGTH = 150;
/** Face crop size expected by EfficientPhys */
export const FACE_SIZE = 72;
/** How many new frames to accumulate before running the next inference */
export const SLIDE_STEP = 30;

export interface IRppgProcessor {
  readonly isReady: boolean;
  readonly shouldInfer: boolean;
  readonly bufferedFrames: number;
  /** Total frames required before first inference (model-specific) */
  readonly warmupFrames: number;
  init(backend: 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm'): Promise<boolean>;
  addFrame(video: HTMLVideoElement, face: FaceResult): ImageData;
  runInference(): Promise<Float32Array | null>;
  getSampleRate(): number;
  reset(): void;
  dispose(): Promise<void>;
}

export class RppgProcessor implements IRppgProcessor {
  private session: ort.InferenceSession | null = null;
  private framesSinceLastInference = 0;
  private isProcessing = false;

  // Pre-allocated ring buffer — INPUT_FRAMES slots of [3, 72, 72] each.
  // Eliminates ~61 KB per-frame GC allocation from the hot path.
  private readonly frameRing: Float32Array[];
  private readonly tsRing: Float64Array;
  private ringHead = 0;   // index of next write (= oldest slot when full)
  private ringCount = 0;  // number of filled slots

  // Pre-allocated inference input tensor storage — avoids 8.9 MB alloc per inference.
  private readonly inputBuffer: Float32Array;

  // Offscreen canvas for face cropping
  private readonly cropCanvas: OffscreenCanvas;
  private readonly cropCtx: OffscreenCanvasRenderingContext2D;

  constructor() {
    this.cropCanvas = new OffscreenCanvas(FACE_SIZE, FACE_SIZE);
    this.cropCtx = this.cropCanvas.getContext('2d', { willReadFrequently: true })!;

    const pixelsPerFrame = 3 * FACE_SIZE * FACE_SIZE; // 15,552
    this.frameRing = Array.from({ length: INPUT_FRAMES }, () => new Float32Array(pixelsPerFrame));
    this.tsRing = new Float64Array(INPUT_FRAMES);
    this.inputBuffer = new Float32Array(INPUT_FRAMES * pixelsPerFrame);
  }

  /**
   * Initialize the EfficientPhys ONNX model.
   */
  async init(backend: 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm' = 'webgpu-high-performance'): Promise<boolean> {
    try {
      const modelUrl = `${import.meta.env.BASE_URL}efficientphys.onnx`;

      if (backend === 'wasm') {
        console.log('[RppgProcessor] Loading model with EP: wasm');
        this.session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
        });
        console.log('%c✅ rPPG model loaded (EfficientPhys) [WASM]', 'color: #ffcc00; font-weight: bold');
      } else {
        const powerPreference = backend === 'webgpu-high-performance' ? 'high-performance' : 'low-power';
        try {
          console.log(`[RppgProcessor] Loading model with EP: webgpu (${powerPreference})`);
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: [{ name: 'webgpu', powerPreference } as any],
          });
          console.log('%c✅ rPPG model loaded (EfficientPhys) [WebGPU]', 'color: #00ff88; font-weight: bold');
        } catch (e) {
          console.warn('[RppgProcessor] WebGPU failed, falling back to WASM:', e);
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
          });
          console.log('%c✅ rPPG model loaded (EfficientPhys) [WASM fallback]', 'color: #ffcc00; font-weight: bold');
        }
      }

      console.log('📥 RppgProcessor inputs:', this.session.inputNames);
      console.log('📤 RppgProcessor outputs:', this.session.outputNames);
      return true;
    } catch (error) {
      console.error('Failed to load rPPG model:', error);
      return false;
    }
  }

  /**
   * Add a video frame to the buffer. Extracts and preprocesses the face ROI.
   * Pixel values are stored as raw [0, 255] floats; standardization is applied
   * at inference time to match the rPPG-Toolbox "Standardized" preprocessing.
   *
   * @param video The live video element
   * @param face Detected face bounding box
   * @returns The face crop as ImageData (for quality monitoring)
   */
  addFrame(video: HTMLVideoElement, face: FaceResult): ImageData {
    const { bbox } = face;

    // Expand bbox by 20% margin for more skin coverage
    const margin = 0.2;
    const mx = bbox.w * margin;
    const my = bbox.h * margin;
    const cropX = Math.max(0, bbox.x - mx);
    const cropY = Math.max(0, bbox.y - my);
    const cropW = Math.min(video.videoWidth - cropX, bbox.w + 2 * mx);
    const cropH = Math.min(video.videoHeight - cropY, bbox.h + 2 * my);

    // Draw face region to the 72x72 crop canvas
    this.cropCtx.drawImage(
      video,
      cropX, cropY, cropW, cropH,   // source rect
      0, 0, FACE_SIZE, FACE_SIZE     // destination rect
    );

    const imageData = this.cropCtx.getImageData(0, 0, FACE_SIZE, FACE_SIZE);

    // Write directly into the pre-allocated ring buffer slot — zero GC.
    const pixelCount = FACE_SIZE * FACE_SIZE;
    const slot = this.frameRing[this.ringHead];
    for (let i = 0; i < pixelCount; i++) {
      slot[i]                   = imageData.data[i * 4];     // R
      slot[pixelCount + i]      = imageData.data[i * 4 + 1]; // G
      slot[2 * pixelCount + i]  = imageData.data[i * 4 + 2]; // B
    }
    this.tsRing[this.ringHead] = performance.now();
    this.ringHead = (this.ringHead + 1) % INPUT_FRAMES;
    if (this.ringCount < INPUT_FRAMES) this.ringCount++;
    this.framesSinceLastInference++;

    return imageData;
  }

  /** Whether we have enough frames to run inference */
  get isReady(): boolean {
    return this.ringCount >= INPUT_FRAMES;
  }

  /** Whether it's time to run the next inference (sliding window advanced enough) */
  get shouldInfer(): boolean {
    return this.isReady && this.framesSinceLastInference >= SLIDE_STEP && !this.isProcessing;
  }

  /** Number of frames buffered so far */
  get bufferedFrames(): number {
    return this.ringCount;
  }

  /** Total frames needed to fill the ring buffer (warmup threshold) */
  get warmupFrames(): number {
    return INPUT_FRAMES;
  }

  /**
   * Compute the actual sample rate from frame timestamps.
   */
  getSampleRate(): number {
    if (this.ringCount < 2) return 30; // default assumption
    const frames = Math.min(this.ringCount, INPUT_FRAMES);
    // Oldest relevant slot — the one written (frames) ago
    const startSlot = (this.ringHead - frames + INPUT_FRAMES) % INPUT_FRAMES;
    const endSlot   = (this.ringHead - 1 + INPUT_FRAMES) % INPUT_FRAMES;
    const elapsed = (this.tsRing[endSlot] - this.tsRing[startSlot]) / 1000;
    return elapsed > 0 ? (frames - 1) / elapsed : 30;
  }

  /**
   * Run EfficientPhys inference on the buffered frames.
   *
   * Input tensor: [1, INPUT_FRAMES, 3, 72, 72] — z-score standardized
   * Output tensor: [1, BVP_LENGTH]
   *
   * @returns BVP signal of length BVP_LENGTH, or null if not ready/error
   */
  async runInference(): Promise<Float32Array | null> {
    if (!this.session || !this.isReady || this.isProcessing) return null;
    this.isProcessing = true;

    try {
      // Assemble contiguous input from ring buffer — no allocation.
      const pixelsPerFrame = 3 * FACE_SIZE * FACE_SIZE;
      const totalSize = INPUT_FRAMES * pixelsPerFrame;
      const inputData = this.inputBuffer; // pre-allocated

      for (let t = 0; t < INPUT_FRAMES; t++) {
        const slot = (this.ringHead + t) % INPUT_FRAMES;
        inputData.set(this.frameRing[slot], t * pixelsPerFrame);
      }

      // Z-score standardization — Welford single-pass mean + M2, then normalize.
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
        [1, INPUT_FRAMES, 3, FACE_SIZE, FACE_SIZE]
      );

      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.session.inputNames[0]] = inputTensor;

      const outputs = await this.session.run(feeds);
      const bvpTensor = outputs[this.session.outputNames[0]];
      // Cast directly — avoids 600-byte Float32Array copy.
      const bvpSignal = bvpTensor.data as Float32Array;

      this.framesSinceLastInference = 0;
      return bvpSignal;
    } catch (error) {
      console.error('rPPG inference error:', error);
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
