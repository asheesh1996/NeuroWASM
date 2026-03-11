// ui.ts
import type { QualityResult } from './qualityMonitor';
import type { SessionStats, FrameUpdate } from './sessionManager';

export class UIManager {
  private readonly canvas: HTMLCanvasElement;
  private readonly ctx: CanvasRenderingContext2D;
  private readonly cameraSelect: HTMLSelectElement;
  private readonly statusBadge: HTMLElement;
  private readonly fpsBadge: HTMLElement;
  private readonly cameraFrame: HTMLElement;
  private readonly cameraFeed: HTMLVideoElement;

  // HR panel elements
  private readonly hrPanel: HTMLElement;
  private readonly hrValue: HTMLElement;
  private readonly qualityFill: HTMLElement;
  private readonly qualityText: HTMLElement;
  private readonly warmupSection: HTMLElement;
  private readonly warmupFill: HTMLElement;
  private readonly warmupText: HTMLElement;
  private readonly elapsedValue: HTMLElement;
  private readonly qualityIssues: HTMLElement;

  // Perf FPS counters
  private readonly faceFpsValue: HTMLElement;
  private readonly hrFpsValue: HTMLElement;

  // Results panel elements
  private readonly resultsPanel: HTMLElement;

  constructor(canvasId: string) {
    this.canvas      = document.getElementById(canvasId) as HTMLCanvasElement;
    this.ctx         = this.canvas.getContext('2d')!;
    this.cameraSelect= document.getElementById('camera-select') as HTMLSelectElement;
    this.statusBadge = document.getElementById('status-badge') as HTMLElement;
    this.fpsBadge    = document.getElementById('fps-badge') as HTMLElement;
    this.cameraFrame = document.getElementById('camera-frame') as HTMLElement;
    this.cameraFeed  = document.getElementById('camera-feed') as HTMLVideoElement;

    // HR panel
    this.hrPanel       = document.getElementById('hr-panel')!;
    this.hrValue       = document.getElementById('hr-value')!;
    this.qualityFill   = document.getElementById('quality-fill')!;
    this.qualityText   = document.getElementById('quality-text')!;
    this.warmupSection = document.getElementById('warmup-section')!;
    this.warmupFill    = document.getElementById('warmup-fill')!;
    this.warmupText    = document.getElementById('warmup-text')!;
    this.elapsedValue  = document.getElementById('elapsed-value')!;
    this.qualityIssues = document.getElementById('quality-issues')!;

    // Perf FPS counters
    this.faceFpsValue = document.getElementById('face-fps-value')!;
    this.hrFpsValue   = document.getElementById('hr-fps-value')!;

    // Results panel
    this.resultsPanel  = document.getElementById('results-panel')!;

    // Resize the camera-frame whenever the window or video dimensions change
    window.addEventListener('resize', () => this.fitFrame());
    this.cameraFeed.addEventListener('loadedmetadata', () => this.fitFrame());
    this.cameraFeed.addEventListener('resize', () => this.fitFrame());
  }

  /**
   * Fit the #camera-frame div to the maximum area that preserves
   * the video's native aspect ratio within the #viewport.
   * The canvas resolution matches the frame size exactly (no CSS scaling).
   */
  fitFrame() {
    const vw = this.cameraFeed.videoWidth  || 1280;
    const vh = this.cameraFeed.videoHeight || 720;
    const viewport = document.getElementById('viewport-content')!;
    const pad = 20 * 2; // 20px padding each side
    // Account for side panel width
    const panelWidth = (!this.hrPanel.hidden || !this.resultsPanel.hidden) ? 320 : 0;
    const maxW = viewport.clientWidth  - pad - panelWidth;
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

  // ── Heart Rate UI ──────────────────────────────────────────

  /** Show or hide the HR measurement panel */
  showHRPanel(visible: boolean) {
    this.hrPanel.hidden = !visible;
    this.fitFrame(); // re-fit camera with/without side panel
  }

  /** Show or hide the results panel */
  showResultsPanel(visible: boolean) {
    this.resultsPanel.hidden = !visible;
    this.fitFrame();
  }

  /** Update all HR panel elements from a FrameUpdate */
  updateFromFrame(update: FrameUpdate) {
    // Heart rate value
    if (update.heartRate > 0) {
      this.hrValue.textContent = String(update.heartRate);
      this.hrValue.classList.add('pulsing');
    } else {
      this.hrValue.textContent = '--';
      this.hrValue.classList.remove('pulsing');
    }

    // Quality bar
    this.updateQuality(update.quality);

    // Warmup progress
    if (update.state === 'warmup') {
      this.warmupSection.hidden = false;
      const pct = Math.round(update.warmupProgress * 100);
      this.warmupFill.style.width = `${pct}%`;
      this.warmupText.textContent = `${pct}%`;
    } else {
      this.warmupSection.hidden = true;
    }

    // Elapsed time
    this.elapsedValue.textContent = this.formatTime(update.elapsed);

    // Per-pipeline FPS counters
    this.faceFpsValue.textContent = update.faceDetectFps > 0 ? update.faceDetectFps.toFixed(1) : '--';
    this.hrFpsValue.textContent   = update.hrInferenceFps > 0 ? update.hrInferenceFps.toFixed(2) : '--';
  }

  /** Update the quality indicator */
  private updateQuality(quality: QualityResult) {
    const pct = Math.round(quality.score * 100);
    this.qualityFill.style.width = `${pct}%`;
    this.qualityFill.className = `quality-fill ${quality.level}`;
    this.qualityText.textContent = quality.level.charAt(0).toUpperCase() + quality.level.slice(1);

    // Show issues
    this.qualityIssues.textContent = quality.issues.length > 0
      ? quality.issues.join(' • ')
      : '';
  }

  /** Display session results */
  showResults(stats: SessionStats) {
    document.getElementById('result-avg-hr')!.textContent = String(stats.avgHR);
    document.getElementById('result-min-hr')!.textContent = String(stats.minHR);
    document.getElementById('result-max-hr')!.textContent = String(stats.maxHR);
    document.getElementById('result-duration')!.textContent = String(Math.round(stats.duration));
    document.getElementById('result-confidence')!.textContent = String(Math.round(stats.avgConfidence * 100));

    this.showHRPanel(false);
    this.showResultsPanel(true);
  }

  /**
   * Draw face detection bounding box on canvas.
   * Color based on quality: green=excellent, yellow=good, red=poor
   */
  drawFaceROI(
    bbox: { x: number; y: number; w: number; h: number },
    videoWidth: number,
    videoHeight: number,
    quality: QualityResult['level']
  ) {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);

    const scaleX = this.canvas.width  / videoWidth;
    const scaleY = this.canvas.height / videoHeight;

    const x = bbox.x * scaleX;
    const y = bbox.y * scaleY;
    const w = bbox.w * scaleX;
    const h = bbox.h * scaleY;

    const colorMap: Record<string, string> = {
      excellent: '#00e8a0',
      good: '#ffcc44',
      poor: '#ff7070',
      invalid: '#555',
    };
    const color = colorMap[quality] ?? '#00e8a0';

    this.ctx.save();

    // Draw face box with glow
    this.ctx.shadowColor = color;
    this.ctx.shadowBlur = 10;
    this.ctx.strokeStyle = color;
    this.ctx.lineWidth = 2;
    this.ctx.setLineDash([8, 4]);
    this.ctx.strokeRect(x, y, w, h);
    this.ctx.setLineDash([]);

    // Draw corner brackets (like a targeting reticle)
    const cornerLen = Math.min(w, h) * 0.2;
    this.ctx.shadowBlur = 6;
    this.ctx.lineWidth = 3;
    this.ctx.setLineDash([]);

    // top-left
    this.ctx.beginPath();
    this.ctx.moveTo(x, y + cornerLen);
    this.ctx.lineTo(x, y);
    this.ctx.lineTo(x + cornerLen, y);
    this.ctx.stroke();

    // top-right
    this.ctx.beginPath();
    this.ctx.moveTo(x + w - cornerLen, y);
    this.ctx.lineTo(x + w, y);
    this.ctx.lineTo(x + w, y + cornerLen);
    this.ctx.stroke();

    // bottom-left
    this.ctx.beginPath();
    this.ctx.moveTo(x, y + h - cornerLen);
    this.ctx.lineTo(x, y + h);
    this.ctx.lineTo(x + cornerLen, y + h);
    this.ctx.stroke();

    // bottom-right
    this.ctx.beginPath();
    this.ctx.moveTo(x + w - cornerLen, y + h);
    this.ctx.lineTo(x + w, y + h);
    this.ctx.lineTo(x + w, y + h - cornerLen);
    this.ctx.stroke();

    this.ctx.restore();
  }

  /** Clear the overlay canvas */
  clearOverlay() {
    this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
  }

  private formatTime(seconds: number): string {
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m}:${s.toString().padStart(2, '0')}`;
  }
}
