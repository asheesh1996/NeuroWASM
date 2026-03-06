// main.ts
import { ModelManager } from './modelManager';
import type { Backend } from './modelManager';
import { CameraManager } from './camera';
import { UIManager } from './ui';

class App {
  private camera: CameraManager;
  private model: ModelManager;
  private ui: UIManager;
  private isRunning = false;
  private frameCount = 0;
  private fpsUpdateTime = 0;

  private get modelName(): string {
    return (document.getElementById('model-select') as HTMLSelectElement).value;
  }
  private get backend(): Backend {
    return (document.getElementById('backend-select') as HTMLSelectElement).value as Backend;
  }

  constructor() {
    this.camera = new CameraManager('camera-feed');
    this.ui = new UIManager('overlay-canvas');
    this.model = new ModelManager(this.modelName, this.backend);
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
      this.ui.setStatus('Running', 'success');
    });

    // Re-initialize model when backend changes
    const backendSelect = document.getElementById('backend-select') as HTMLSelectElement;
    backendSelect.addEventListener('change', async () => {
      this.ui.setStatus('Loading Model...', 'warning');
      await this.model.dispose();
      this.model = new ModelManager(this.modelName, this.backend);
      const ok = await this.model.initialize();
      if (ok) {
        this.ui.setModelManager(this.model);
        this.ui.setStatus('Running', 'success');
      } else {
        this.ui.setStatus('Model Load Failed', 'danger');
      }
    });

    await this.loadModel();

    this.isRunning = true;
    this.fpsUpdateTime = performance.now();
    this.processFrame();
  }

  private async loadModel() {
    this.ui.setStatus('Loading Model...', 'warning');
    const ok = await this.model.initialize();
    if (!ok) {
      this.ui.setStatus('Model Load Failed', 'danger');
      return;
    }
    this.ui.setModelManager(this.model);
    this.ui.setStatus('Running', 'success');
  }

  private processFrame() {
    if (!this.isRunning) return;
    // Serialised loop: await each inference before requesting the next frame.
    // Prevents WebGPU "buffer unmapped" crashes caused by concurrent OrtRun calls
    // when rAF fires faster than the GPU finishes the previous async inference.
    const loop = async () => {
      while (this.isRunning) {
        await new Promise<number>(resolve => requestAnimationFrame(resolve));

        const video = this.camera.getVideoElement();
        if (video.readyState >= video.HAVE_ENOUGH_DATA && video.videoWidth > 0) {
          const boxes = await this.model.runInference(video);
          this.ui.drawBoxes(boxes, video.videoWidth, video.videoHeight);

          this.frameCount++;
          const now = performance.now();
          if (now - this.fpsUpdateTime >= 1000) {
            this.ui.setFPS((this.frameCount * 1000) / (now - this.fpsUpdateTime));
            this.frameCount = 0;
            this.fpsUpdateTime = now;
          }
        }
      }
    };
    loop().catch(console.error);
  }
}

document.addEventListener('DOMContentLoaded', () => {
  new App().initialize().catch(console.error);
});
