// ui.ts
import type { BoundingBox, ModelManager } from './modelManager';

export class UIManager {
  private canvas: HTMLCanvasElement;
  private ctx: CanvasRenderingContext2D;
  private cameraSelect: HTMLSelectElement;
  private statusBadge: HTMLElement;
  private fpsBadge: HTMLElement;
  private cameraFrame: HTMLElement;
  private cameraFeed: HTMLVideoElement;
  private modelManager: ModelManager | null = null;

  constructor(canvasId: string) {
    this.canvas      = document.getElementById(canvasId) as HTMLCanvasElement;
    this.ctx         = this.canvas.getContext('2d')!;
    this.cameraSelect= document.getElementById('camera-select') as HTMLSelectElement;
    this.statusBadge = document.getElementById('status-badge') as HTMLElement;
    this.fpsBadge    = document.getElementById('fps-badge') as HTMLElement;
    this.cameraFrame = document.getElementById('camera-frame') as HTMLElement;
    this.cameraFeed  = document.getElementById('camera-feed') as HTMLVideoElement;

    // Resize the camera-frame whenever the window or video dimensions change
    window.addEventListener('resize', () => this.fitFrame());
    this.cameraFeed.addEventListener('loadedmetadata', () => this.fitFrame());
    this.cameraFeed.addEventListener('resize', () => this.fitFrame());
  }

  setModelManager(mm: ModelManager) {
    this.modelManager = mm;
  }

  /**
   * Fit the #camera-frame div to the maximum area that preserves
   * the video's native aspect ratio within the #viewport.
   * The canvas resolution matches the frame size exactly (no CSS scaling).
   */
  fitFrame() {
    const vw = this.cameraFeed.videoWidth  || 1280;
    const vh = this.cameraFeed.videoHeight || 720;
    const viewport = document.getElementById('viewport')!;
    const pad = 20 * 2; // 20px padding each side
    const maxW = viewport.clientWidth  - pad;
    const maxH = viewport.clientHeight - pad;

    const scale = Math.min(maxW / vw, maxH / vh);
    const frameW = Math.round(vw * scale);
    const frameH = Math.round(vh * scale);

    this.cameraFrame.style.width  = `${frameW}px`;
    this.cameraFrame.style.height = `${frameH}px`;

    // Canvas resolution = frame pixel size → 1:1 mapping with video coords
    this.canvas.width  = frameW;
    this.canvas.height = frameH;
  }

  populateCameras(devices: MediaDeviceInfo[]) {
    this.cameraSelect.innerHTML = '';
    if (devices.length === 0) {
      const opt = document.createElement('option');
      opt.text = 'No cameras found';
      opt.disabled = true;
      opt.selected = true;
      this.cameraSelect.add(opt);
      return;
    }
    devices.forEach((device, index) => {
      const opt = document.createElement('option');
      opt.value = device.deviceId;
      opt.text  = device.label || `Camera ${index + 1}`;
      this.cameraSelect.add(opt);
    });
  }

  onCameraChange(callback: (deviceId: string) => void) {
    this.cameraSelect.addEventListener('change', (e) =>
      callback((e.target as HTMLSelectElement).value)
    );
  }

  setStatus(message: string, type: 'primary' | 'success' | 'danger' | 'warning') {
    const map: Record<string, string> = {
      primary: 'initializing',
      success: 'success',
      danger:  'danger',
      warning: 'warning',
    };
    this.statusBadge.className = `status-pill ${map[type] ?? 'initializing'}`;
    this.statusBadge.textContent = message;
  }

  setFPS(fps: number) {
    this.fpsBadge.classList.add('visible');
    this.fpsBadge.textContent = `${fps.toFixed(1)} FPS`;
  }

  /**
   * Draw bounding boxes.
   * Because the video uses object-fit:fill inside the frame and the canvas
   * has the same pixel dimensions as the frame, we simply scale video
   * coordinates by (canvasW / videoW, canvasH / videoH).
   */
  drawBoxes(boxes: BoundingBox[], videoWidth: number, videoHeight: number) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    if (boxes.length === 0) return;
    if (!videoWidth || !videoHeight) return;

    const scaleX = this.canvas.width  / videoWidth;
    const scaleY = this.canvas.height / videoHeight;

    this.ctx.save();

    for (const box of boxes) {
      const x = box.x1 * scaleX;
      const y = box.y1 * scaleY;
      const w = (box.x2 - box.x1) * scaleX;
      const h = (box.y2 - box.y1) * scaleY;
      const color = this.modelManager?.getClassColor(box.classIndex) ?? '#00ff90';
      const label = `${box.label} ${(box.score * 100).toFixed(0)}%`;

      // Box glow
      this.ctx.shadowColor = color;
      this.ctx.shadowBlur  = 8;
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth   = 2;
      this.ctx.strokeRect(x, y, w, h);

      // Label pill
      this.ctx.shadowBlur  = 0;
      this.ctx.font        = 'bold 12px Inter, Arial, sans-serif';
      const tw   = this.ctx.measureText(label).width + 10;
      const th   = 20;
      const ly   = y > th + 4 ? y - th - 2 : y + 2;

      this.ctx.fillStyle = color;
      this.ctx.beginPath();
      this.ctx.roundRect(x - 1, ly, tw, th, 4);
      this.ctx.fill();

      this.ctx.fillStyle  = '#ffffff';
      this.ctx.shadowColor= 'rgba(0,0,0,0.6)';
      this.ctx.shadowBlur = 2;
      this.ctx.fillText(label, x + 4, ly + 14);
    }

    this.ctx.restore();
  }
}
