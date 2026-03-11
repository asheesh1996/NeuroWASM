// main.ts
import * as ort from 'onnxruntime-web/webgpu';
import { SessionManager } from './sessionManager';
import type { SessionState } from './sessionManager';
import { CameraManager } from './camera';
import { UIManager } from './ui';
import { FaceDetector } from './faceDetector';
import type { IFaceDetector } from './faceDetector';
import { RppgProcessor } from './rppgProcessor';
import type { IRppgProcessor } from './rppgProcessor';
import { MediaPipeDetector } from './mediapipeDetector';
import { FactorizePhysProcessor } from './factorizePhysProcessor';

// Point ONNX Runtime WASM binaries to the CDN
// The WebGPU EP still loads .mjs + .wasm bootstrap files internally;
// the CDN path avoids issues with Vite's strict public-dir serving rules.
ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';

/**
 * Probe WebGPU availability and log adapter details.
 * Returns true if a WebGPU adapter is available.
 */
async function setupWebGPU(): Promise<boolean> {
  if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
    console.log('[WebGPU] navigator.gpu not available');
    return false;
  }

  let adapter: any = null;
  try {
    adapter = await (navigator as any).gpu.requestAdapter({ powerPreference: 'high-performance' });
  } catch (e) {
    console.warn('[WebGPU] requestAdapter() threw:', e);
    return false;
  }
  if (!adapter) {
    console.log('[WebGPU] requestAdapter() returned null — no GPU available');
    return false;
  }

  // Log adapter info
  try {
    const info = await adapter.requestAdapterInfo();
    console.log(
      `%c🎮 WebGPU Adapter: ${info.vendor ?? '?'} — ${info.device ?? info.description ?? 'unknown'}`,
      'color: #00ff88; font-weight: bold',
    );
  } catch {
    if ('info' in adapter) {
      const ai = adapter.info;
      console.log('[WebGPU] Adapter:', ai.vendor, ai.architecture, ai.device || ai.description);
    }
  }
  console.log('[WebGPU] Max buffer size:', adapter.limits.maxBufferSize);
  console.log('[WebGPU] Max storage buffer binding size:', adapter.limits.maxStorageBufferBindingSize);

  return true;
}

type Backend = 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm';

function createFaceDetector(id: string): IFaceDetector {
  if (id === 'blazeface') return new MediaPipeDetector();
  return new FaceDetector();
}

function createRppgProcessor(id: string): IRppgProcessor {
  if (id === 'factorizephys') return new FactorizePhysProcessor();
  return new RppgProcessor();
}

class App {
  private readonly camera: CameraManager;
  private session!: SessionManager;
  private readonly ui: UIManager;
  private isRunning = false;
  private frameCount = 0;
  private fpsUpdateTime = 0;

  private get backend(): Backend {
    return (document.getElementById('backend-select') as HTMLSelectElement).value as Backend;
  }

  private get rppgModelId(): string {
    return (document.getElementById('rppg-model-select') as HTMLSelectElement).value;
  }

  private get faceModelId(): string {
    return (document.getElementById('face-model-select') as HTMLSelectElement).value;
  }

  private readonly measureBtn: HTMLButtonElement;
  private readonly resetBtn: HTMLButtonElement;
  private isInferring = false;

  constructor() {
    this.camera = new CameraManager('camera-feed');
    this.ui = new UIManager('overlay-canvas');
    this.measureBtn = document.getElementById('measure-btn') as HTMLButtonElement;
    this.resetBtn = document.getElementById('reset-btn') as HTMLButtonElement;

    // Wire up the WebGPU banner dismiss button
    document.getElementById('webgpu-banner-close')!.addEventListener('click', () => {
      document.getElementById('webgpu-banner')!.hidden = true;
    });
  }

  async initialize() {
    this.ui.setStatus('Requesting Camera...', 'warning');

    const devices = await this.camera.getAvailableCameras();
    this.ui.populateCameras(devices);

    const started = devices.length > 0
      ? await this.camera.startCamera(devices[0].deviceId)
      : await this.camera.startCamera();

    if (!started) {
      this.ui.setStatus('Camera Failed', 'danger');
      return;
    }

    this.ui.onCameraChange(async (deviceId) => {
      this.ui.setStatus('Switching Camera...', 'warning');
      await this.camera.startCamera(deviceId);
      this.ui.setStatus('Ready', 'success');
    });

    // Re-initialize models when backend changes
    const backendSelect = document.getElementById('backend-select') as HTMLSelectElement;
    backendSelect.addEventListener('change', async () => {
      if (this.session.state !== 'idle' && this.session.state !== 'complete') return; // don't switch during measurement
      this.ui.setStatus('Loading Models...', 'warning');
      await this.session.dispose();
      await this.loadModels();
    });

    // Re-initialize when rPPG or face model is changed
    const rppgModelSelect = document.getElementById('rppg-model-select') as HTMLSelectElement;
    const faceModelSelect = document.getElementById('face-model-select') as HTMLSelectElement;
    const handleModelChange = async () => {
      const isActive = this.session.state === 'warmup' || this.session.state === 'measuring';
      if (isActive) {
        this.session.forceStop();
        this.syncMeasureButtons('Start Measurement', false, false);
        this.resetBtn.hidden = true;
        this.ui.showHRPanel(false);
        this.ui.clearOverlay();
      }
      this.ui.setStatus('Loading Models...', 'warning');
      this.measureBtn.disabled = true;
      await this.session.dispose();
      await this.loadModels();
    };
    rppgModelSelect.addEventListener('change', handleModelChange);
    faceModelSelect.addEventListener('change', handleModelChange);

    // Wire up measure buttons
    this.measureBtn.addEventListener('click', () => this.toggleMeasurement());
    this.resetBtn.addEventListener('click', () => this.resetMeasurement());

    // Load models
    await this.loadModels();

    // Start the frame loop
    this.isRunning = true;
    this.fpsUpdateTime = performance.now();
    this.processFrame();
  }

  private async loadModels() {
    this.ui.setStatus('Loading Models...', 'warning');

    // Resolve the effective backend by probing/setting up WebGPU
    let backend = this.backend;
    const webgpuBanner = document.getElementById('webgpu-banner')!;
    if (backend !== 'wasm') {
      const gpuReady = await setupWebGPU();
      if (!gpuReady) {
        console.warn('[App] WebGPU not usable, switching to WASM backend');
        backend = 'wasm';
        (document.getElementById('backend-select') as HTMLSelectElement).value = 'wasm';
        webgpuBanner.hidden = false;
      } else {
        webgpuBanner.hidden = true;
      }
    }

    const fd = createFaceDetector(this.faceModelId);
    const rp = createRppgProcessor(this.rppgModelId);
    this.session = new SessionManager(fd, rp);

    const ok = await this.session.init(backend);
    if (!ok) {
      this.ui.setStatus('Model Load Failed', 'danger');
      return;
    }
    this.ui.setStatus('Ready', 'success');
    this.measureBtn.disabled = false;
  }

  /** Sync both measure buttons (results panel + HR panel) to the same visual state */
  private syncMeasureButtons(text: string, active: boolean, disabled: boolean) {
    this.measureBtn.textContent = text;
    this.measureBtn.classList.toggle('active', active);
    this.measureBtn.disabled = disabled;
  }

  private toggleMeasurement() {
    const state = this.session.state;

    if (state === 'idle' || state === 'complete') {
      // Start new measurement
      this.session.start();
      this.syncMeasureButtons('Stop Measurement', true, false);
      this.resetBtn.hidden = false;
      this.ui.showResultsPanel(false);
      this.ui.showHRPanel(true);
      this.ui.setStatus('Measuring', 'success');
    } else if (state === 'warmup' || state === 'measuring') {
      // Stop measurement
      if (this.session.canStop) {
        const stats = this.session.stop();
        if (stats) this.showCompletedState(stats);
      } else {
        const stats = this.session.forceStop();
        if (stats) this.showCompletedState(stats);
      }
    }
  }

  /** Reset (restart) the measurement from 0 seconds */
  private resetMeasurement() {
    // Force-stop any running session, then immediately start fresh
    if (this.session.state === 'warmup' || this.session.state === 'measuring') {
      this.session.forceStop();
    }
    this.session.start();
    this.syncMeasureButtons('Stop Measurement', true, false);
    this.resetBtn.hidden = false;
    this.ui.showResultsPanel(false);
    this.ui.showHRPanel(true);
    this.ui.setStatus('Measuring', 'success');
  }

  private showCompletedState(stats: { avgHR: number; minHR: number; maxHR: number; duration: number; avgConfidence: number; hrHistory: number[] }) {
    this.ui.showResults(stats);
    this.syncMeasureButtons('Start Measurement', false, false);
    this.resetBtn.hidden = true;
    this.ui.setStatus('Complete', 'success');
    this.ui.clearOverlay();
  }

  private async handleActiveFrame(video: HTMLVideoElement) {
    const update = await this.session.processFrame(video);
    this.ui.updateFromFrame(update);

    if (update.face) {
      this.ui.drawFaceROI(update.face.bbox, video.videoWidth, video.videoHeight, update.quality.level);
    } else {
      this.ui.clearOverlay();
    }

    if (this.session.canStop) {
      this.syncMeasureButtons('Stop Measurement', true, false);
    } else {
      const remaining = Math.ceil(SessionManager.MIN_SESSION_SECONDS - update.elapsed);
      this.syncMeasureButtons(`Stop (${remaining}s)`, true, false);
    }

    // Auto-complete when session reaches the minimum duration
    if (this.session.canStop && update.state === 'measuring') {
      const stats = this.session.stop();
      if (stats) this.showCompletedState(stats);
    }
  }

  private processFrame() {
    if (!this.isRunning) return;

    const loop = async () => {
      while (this.isRunning) {
        await new Promise<number>(resolve => requestAnimationFrame(resolve));

        const video = this.camera.getVideoElement();
        if (video.readyState < video.HAVE_ENOUGH_DATA || video.videoWidth === 0) continue;

        const state: SessionState = this.session.state;
        if (state === 'warmup' || state === 'measuring') {
          // Fire-and-forget: decouple inference from the rAF tick so the
          // animation loop is never blocked by a 50-300 ms ONNX dispatch.
          if (!this.isInferring) {
            this.isInferring = true;
            this.handleActiveFrame(video).finally(() => { this.isInferring = false; });
          }
        } else {
          this.ui.clearOverlay();
        }

        this.frameCount++;
        const now = performance.now();
        if (now - this.fpsUpdateTime >= 1000) {
          this.ui.setFPS((this.frameCount * 1000) / (now - this.fpsUpdateTime));
          this.frameCount = 0;
          this.fpsUpdateTime = now;
        }
      }
    };
    loop().catch(console.error);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new App().initialize().catch(console.error);
});
