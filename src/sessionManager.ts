// sessionManager.ts
// Orchestrates rPPG measurement sessions: state machine, timing, result aggregation.

import type { IFaceDetector, FaceResult } from './faceDetector';
import type { IRppgProcessor } from './rppgProcessor';
import { estimateHeartRate, HRSmoother } from './signalProcessing';
import type { HREstimate } from './signalProcessing';
import { QualityMonitor } from './qualityMonitor';
import type { QualityResult } from './qualityMonitor';

export type SessionState = 'idle' | 'warmup' | 'measuring' | 'complete';

export interface SessionStats {
  avgHR: number;
  minHR: number;
  maxHR: number;
  duration: number;  // seconds
  avgConfidence: number;
  hrHistory: number[];
}

export interface FrameUpdate {
  /** Current session state */
  state: SessionState;
  /** Detected face (if any) */
  face: FaceResult | null;
  /** Signal quality assessment */
  quality: QualityResult;
  /** Current smoothed heart rate (0 if not available) */
  heartRate: number;
  /** Current confidence score */
  confidence: number;
  /** Warmup progress 0-1 */
  warmupProgress: number;
  /** Elapsed measurement time in seconds */
  elapsed: number;
  /** Face detection throughput in Hz (EWMA) */
  faceDetectFps: number;
  /** HR inference throughput in Hz (EWMA) */
  hrInferenceFps: number;
}

export class SessionManager {
  private readonly faceDetector: IFaceDetector;
  private readonly rppgProcessor: IRppgProcessor;
  private readonly qualityMonitor: QualityMonitor;
  private readonly hrSmoother: HRSmoother;

  private _state: SessionState = 'idle';
  private startTime = 0;
  private hrHistory: number[] = [];
  private confidenceHistory: number[] = [];
  private currentHR = 0;
  private currentConfidence = 0;
  private lastQuality: QualityResult = { score: 0, level: 'invalid', issues: [] };

  // Face-detection throttle: only dispatch to ONNX at ~10 Hz instead of every rAF frame.
  private lastFaceTime = 0;
  private lastFace: FaceResult | null = null;

  // Per-pipeline EWMA FPS counters (updated via inter-event interval).
  private faceDetectFps = 0;
  private prevFaceDetectTime = 0;
  private hrInferenceFps = 0;
  private prevHrInferenceTime = 0;

  static readonly MIN_SESSION_SECONDS = 60;

  get state(): SessionState { return this._state; }
  /** Elapsed measurement time (warmup + measuring) in seconds */
  get elapsed(): number {
    if (this.startTime === 0) return 0;
    return (performance.now() - this.startTime) / 1000;
  }
  /** Whether the session has met the minimum duration */
  get canStop(): boolean {
    return this.elapsed >= SessionManager.MIN_SESSION_SECONDS;
  }

  constructor(faceDetector: IFaceDetector, rppgProcessor: IRppgProcessor) {
    this.faceDetector = faceDetector;
    this.rppgProcessor = rppgProcessor;
    this.qualityMonitor = new QualityMonitor();
    this.hrSmoother = new HRSmoother();
  }

  /**
   * Initialize both models.
   */
  async init(backend: 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm' = 'webgpu-high-performance'): Promise<boolean> {
    // Load sequentially to avoid WASM double-init race condition
    const faceOk = await this.faceDetector.init(backend);
    const rppgOk = await this.rppgProcessor.init(backend);
    return faceOk && rppgOk;
  }

  /**
   * Start a new measurement session. Transitions from idle → warmup.
   */
  start(): void {
    if (this._state !== 'idle' && this._state !== 'complete') return;
    this.rppgProcessor.reset();
    this.qualityMonitor.reset();
    this.hrSmoother.reset();
    this.hrHistory = [];
    this.confidenceHistory = [];
    this.currentHR = 0;
    this.currentConfidence = 0;
    this.startTime = performance.now();
    this.lastFaceTime = 0;
    this.lastFace = null;
    this.faceDetectFps = 0;
    this.prevFaceDetectTime = 0;
    this.hrInferenceFps = 0;
    this.prevHrInferenceTime = 0;
    this._state = 'warmup';
  }

  /**
   * Stop the current session. Only allowed after MIN_SESSION_SECONDS.
   * @returns Final session statistics
   */
  stop(): SessionStats | null {
    if (this._state !== 'measuring') return null;

    this._state = 'complete';
    return this.getResults();
  }

  /**
   * Force stop the session regardless of duration.
   */
  forceStop(): SessionStats | null {
    if (this._state === 'idle' || this._state === 'complete') return null;
    this._state = 'complete';
    return this.getResults();
  }

  /**
   * Process a single video frame. Call this every frame in the animation loop.
   *
   * This method orchestrates:
   * 1. Face detection
   * 2. Quality monitoring
   * 3. Frame buffering
   * 4. rPPG inference (when sliding window advances)
   * 5. Signal processing → HR estimate
   */
  async processFrame(video: HTMLVideoElement): Promise<FrameUpdate> {
    const baseUpdate: FrameUpdate = {
      state: this._state,
      face: null,
      quality: this.lastQuality,
      heartRate: this.currentHR,
      confidence: this.currentConfidence,
      warmupProgress: 0,
      elapsed: this.elapsed,
      faceDetectFps: 0,
      hrInferenceFps: 0,
    };

    if (this._state === 'idle' || this._state === 'complete') {
      return baseUpdate;
    }

    // 1. Face detection — throttled to ~10 Hz to reduce GPU dispatch pressure.
    const now = performance.now();
    if (now - this.lastFaceTime >= 100) {
      this.lastFace = await this.faceDetector.detect(video);
      this.lastFaceTime = now;
      if (this.prevFaceDetectTime > 0) {
        const interval = now - this.prevFaceDetectTime;
        this.faceDetectFps = this.faceDetectFps * 0.7 + (1000 / interval) * 0.3;
      }
      this.prevFaceDetectTime = now;
    }
    const face = this.lastFace;
    baseUpdate.face = face;

    // 2. Get face ROI for quality check
    let faceROI: ImageData | null = null;
    if (face) {
      faceROI = this.rppgProcessor.addFrame(video, face);
    }

    // 3. Quality monitoring
    const quality = this.qualityMonitor.update(
      faceROI,
      face?.bbox ?? null,
      video.videoWidth,
      video.videoHeight,
    );
    this.lastQuality = quality;
    baseUpdate.quality = quality;

    // 4. Warmup progress
    const warmupProgress = Math.min(1, this.rppgProcessor.bufferedFrames / this.rppgProcessor.warmupFrames);
    baseUpdate.warmupProgress = warmupProgress;

    // Transition warmup → measuring when buffer is ready
    if (this._state === 'warmup' && this.rppgProcessor.isReady) {
      this._state = 'measuring';
      baseUpdate.state = 'measuring';
    }

    // 5. rPPG inference when sliding window advances
    if (this._state === 'measuring' && this.rppgProcessor.shouldInfer && quality.level !== 'invalid') {
      const bvp = await this.rppgProcessor.runInference();
      if (bvp) {
        const sampleRate = this.rppgProcessor.getSampleRate();
        const estimate: HREstimate = estimateHeartRate(bvp, sampleRate);

        const smoothedHR = this.hrSmoother.update(estimate.hr, estimate.confidence);
        this.currentHR = smoothedHR;
        this.currentConfidence = estimate.confidence;

        if (smoothedHR > 0) {
          this.hrHistory.push(smoothedHR);
          this.confidenceHistory.push(estimate.confidence);
        }

        const hrNow = performance.now();
        if (this.prevHrInferenceTime > 0) {
          const interval = hrNow - this.prevHrInferenceTime;
          this.hrInferenceFps = this.hrInferenceFps * 0.7 + (1000 / interval) * 0.3;
        }
        this.prevHrInferenceTime = hrNow;
      }
    }

    baseUpdate.heartRate = this.currentHR;
    baseUpdate.confidence = this.currentConfidence;
    baseUpdate.elapsed = this.elapsed;
    baseUpdate.faceDetectFps = this.faceDetectFps;
    baseUpdate.hrInferenceFps = this.hrInferenceFps;

    return baseUpdate;
  }

  /**
   * Compute session statistics.
   */
  getResults(): SessionStats {
    const valid = this.hrHistory.filter(h => h > 0);
    if (valid.length === 0) {
      return { avgHR: 0, minHR: 0, maxHR: 0, duration: this.elapsed, avgConfidence: 0, hrHistory: [] };
    }

    const avgHR = Math.round(valid.reduce((a, b) => a + b, 0) / valid.length);
    const minHR = Math.round(Math.min(...valid));
    const maxHR = Math.round(Math.max(...valid));
    const avgConfidence = this.confidenceHistory.length > 0
      ? this.confidenceHistory.reduce((a, b) => a + b, 0) / this.confidenceHistory.length
      : 0;

    return {
      avgHR,
      minHR,
      maxHR,
      duration: this.elapsed,
      avgConfidence,
      hrHistory: [...valid],
    };
  }

  async dispose(): Promise<void> {
    await Promise.all([
      this.faceDetector.dispose(),
      this.rppgProcessor.dispose(),
    ]);
  }
}
