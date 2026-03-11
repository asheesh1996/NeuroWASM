// faceDetector.ts
// UltraFace-320 ONNX-based lightweight face detection for rPPG ROI extraction.

import * as ort from 'onnxruntime-web/webgpu';

export interface FaceResult {
  /** Bounding box in video coordinates */
  bbox: { x: number; y: number; w: number; h: number };
  /** Detection confidence 0-1 */
  confidence: number;
  /** Whether the face position is stable (not moving much) */
  stable: boolean;
}

export interface IFaceDetector {
  init(backend: 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm'): Promise<boolean>;
  detect(video: HTMLVideoElement): Promise<FaceResult | null>;
  dispose(): Promise<void>;
}

export class FaceDetector implements IFaceDetector {
  private session: ort.InferenceSession | null = null;
  private readonly inputWidth = 320;
  private readonly inputHeight = 240;
  private prevCenter: { x: number; y: number } | null = null;
  private readonly stabilityThreshold = 12; // px movement considered "unstable"

  // Offscreen canvas for preprocessing
  private readonly preprocessCanvas: OffscreenCanvas;
  private readonly preprocessCtx: OffscreenCanvasRenderingContext2D;

  // Pre-allocated pixel buffer — reused every frame to avoid GC pressure (~900 KB)
  private readonly preprocessBuffer: Float32Array;
  // Pre-allocated feeds object — key assigned once after model init
  private readonly feeds: Record<string, ort.Tensor> = {};

  constructor() {
    this.preprocessCanvas = new OffscreenCanvas(this.inputWidth, this.inputHeight);
    this.preprocessCtx = this.preprocessCanvas.getContext('2d', { willReadFrequently: true })!;
    this.preprocessBuffer = new Float32Array(3 * this.inputWidth * this.inputHeight);
  }

  /**
   * Initialize the face detection model.
   * @param backend ONNX Runtime execution provider
   */
  async init(backend: 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm' = 'webgpu-high-performance'): Promise<boolean> {
    try {
      const modelUrl = `${import.meta.env.BASE_URL}ultraface_320.onnx`;

      if (backend === 'wasm') {
        console.log('[FaceDetector] Loading model with EP: wasm');
        this.session = await ort.InferenceSession.create(modelUrl, {
          executionProviders: ['wasm'],
        });
        console.log('%c✅ Face detector loaded (UltraFace-320) [WASM]', 'color: #ffcc00; font-weight: bold');
      } else {
        const powerPreference = backend === 'webgpu-high-performance' ? 'high-performance' : 'low-power';
        try {
          console.log(`[FaceDetector] Loading model with EP: webgpu (${powerPreference})`);
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: [{ name: 'webgpu', powerPreference } as any],
          });
          console.log('%c✅ Face detector loaded (UltraFace-320) [WebGPU]', 'color: #00ff88; font-weight: bold');
        } catch (e) {
          console.warn('[FaceDetector] WebGPU failed, falling back to WASM:', e);
          this.session = await ort.InferenceSession.create(modelUrl, {
            executionProviders: ['wasm'],
          });
          console.log('%c✅ Face detector loaded (UltraFace-320) [WASM fallback]', 'color: #ffcc00; font-weight: bold');
        }
      }

      console.log('📥 FaceDetector inputs:', this.session.inputNames);
      console.log('📤 FaceDetector outputs:', this.session.outputNames);
      return true;
    } catch (error) {
      console.error('Failed to load face detector:', error);
      return false;
    }
  }

  /**
   * Detect the largest face in the video frame.
   * @returns FaceResult with bbox in original video coordinates, or null if no face.
   */
  async detect(video: HTMLVideoElement): Promise<FaceResult | null> {
    if (!this.session) return null;

    const vw = video.videoWidth;
    const vh = video.videoHeight;
    if (vw === 0 || vh === 0) return null;

    // Preprocess: resize to 320x240
    this.preprocessCtx.drawImage(video, 0, 0, this.inputWidth, this.inputHeight);
    const imageData = this.preprocessCtx.getImageData(0, 0, this.inputWidth, this.inputHeight);

    // Convert to NCHW Float32, normalize per UltraFace spec: (pixel - 127) / 128
    // Reuse pre-allocated buffer to eliminate ~900 KB per-frame GC allocation
    const pixelCount = this.inputWidth * this.inputHeight;
    const float32 = this.preprocessBuffer;
    for (let i = 0; i < pixelCount; i++) {
      float32[i]                   = (imageData.data[i * 4]     - 127) / 128; // R
      float32[pixelCount + i]      = (imageData.data[i * 4 + 1] - 127) / 128; // G
      float32[2 * pixelCount + i]  = (imageData.data[i * 4 + 2] - 127) / 128; // B
    }

    const inputTensor = new ort.Tensor('float32', float32, [1, 3, this.inputHeight, this.inputWidth]);
    this.feeds[this.session.inputNames[0]] = inputTensor;

    const outputs = await this.session.run(this.feeds);

    // UltraFace outputs:
    // "scores": [1, N, 2] — [background_prob, face_prob]
    // "boxes":  [1, N, 4] — [x1, y1, x2, y2] normalized to [0, 1]
    const scoresName = this.session.outputNames.find((n: string) => n.includes('score')) ?? this.session.outputNames[0];
    const boxesName = this.session.outputNames.find((n: string) => n.includes('box')) ?? this.session.outputNames[1];

    const scores = outputs[scoresName].data as Float32Array;
    const boxes = outputs[boxesName].data as Float32Array;
    const numDetections = outputs[scoresName].dims[1] as number;

    // Find the highest-confidence face detection
    let bestScore = 0;
    let bestIdx = -1;
    const confThreshold = 0.7;

    for (let i = 0; i < numDetections; i++) {
      const faceProb = scores[i * 2 + 1]; // face probability
      if (faceProb > bestScore && faceProb > confThreshold) {
        bestScore = faceProb;
        bestIdx = i;
      }
    }

    if (bestIdx < 0) {
      this.prevCenter = null;
      return null;
    }

    // Convert normalized [0,1] box coords to video pixel coordinates
    const bx1 = boxes[bestIdx * 4 + 0] * vw;
    const by1 = boxes[bestIdx * 4 + 1] * vh;
    const bx2 = boxes[bestIdx * 4 + 2] * vw;
    const by2 = boxes[bestIdx * 4 + 3] * vh;

    const bbox = {
      x: Math.max(0, bx1),
      y: Math.max(0, by1),
      w: Math.min(vw - bx1, bx2 - bx1),
      h: Math.min(vh - by1, by2 - by1),
    };

    // Check stability
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
