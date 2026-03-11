// qualityMonitor.ts
// Monitors signal quality conditions that affect rPPG accuracy.

export interface QualityResult {
  /** Overall quality score 0-1 */
  score: number;
  /** Human-readable quality level */
  level: 'excellent' | 'good' | 'poor' | 'invalid';
  /** Active quality issues */
  issues: string[];
}

export interface FaceBBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

export class QualityMonitor {
  private prevBBox: FaceBBox | null = null;
  private framesWithoutFace = 0;

  // Ring buffers — eliminate O(n) Array.shift() on every frame.
  private readonly motionRing = new Float64Array(30);
  private motionHead = 0;
  private motionCount = 0;

  private readonly tsRing = new Float64Array(60);
  private tsHead = 0;
  private tsCount = 0;

  // Thresholds
  private readonly MIN_BRIGHTNESS = 50;
  private readonly MAX_BRIGHTNESS = 220;
  private readonly MAX_MOTION_PX = 15;
  private readonly MIN_FACE_RATIO = 0.03;   // 3% of frame
  private readonly MAX_FACE_RATIO = 0.65;   // 65% of frame
  private readonly MIN_FPS = 12;

  /**
   * Update quality metrics with the latest frame data.
   *
   * @param faceROI Face crop ImageData (for brightness check), or null if no face
   * @param bbox Current face bounding box in video coordinates, or null
   * @param frameWidth Video frame width
   * @param frameHeight Video frame height
   */
  update(
    faceROI: ImageData | null,
    bbox: FaceBBox | null,
    frameWidth: number,
    frameHeight: number,
  ): QualityResult {
    const issues: string[] = [];
    let score = 1;

    // Track frame timestamps for FPS calculation (ring buffer).
    const now = performance.now();
    this.tsRing[this.tsHead] = now;
    this.tsHead = (this.tsHead + 1) % 60;
    if (this.tsCount < 60) this.tsCount++;

    // ── Face presence ──
    score -= this.checkFacePresence(bbox, issues);

    // Early exit if face lost too long
    if (this.framesWithoutFace > 5) {
      return { score: 0, level: 'invalid', issues: ['No face detected'] };
    }

    // ── Lighting (mean brightness of face ROI) ──
    score -= this.checkLighting(faceROI, issues);

    // ── Face size ──
    score -= this.checkFaceSize(bbox, frameWidth, frameHeight, issues);

    // ── Motion (face bbox displacement) ──
    score -= this.checkMotion(bbox, issues);

    if (bbox) this.prevBBox = { ...bbox };

    // ── Frame rate ──
    score -= this.checkFPS(issues);

    // Clamp score
    score = Math.max(0, Math.min(1, score));

    return { score, level: this.scoreToLevel(score), issues };
  }

  private scoreToLevel(score: number): QualityResult['level'] {
    if (score >= 0.8) return 'excellent';
    if (score >= 0.5) return 'good';
    if (score > 0) return 'poor';
    return 'invalid';
  }

  private checkFacePresence(bbox: FaceBBox | null, issues: string[]): number {
    if (bbox) {
      this.framesWithoutFace = 0;
      return 0;
    }
    this.framesWithoutFace++;
    issues.push('Face lost momentarily');
    return 0.3;
  }

  private checkLighting(faceROI: ImageData | null, issues: string[]): number {
    if (!faceROI) return 0;
    const brightness = this.meanBrightness(faceROI);
    if (brightness < this.MIN_BRIGHTNESS) {
      issues.push('Too dark');
      return 0.3;
    }
    if (brightness > this.MAX_BRIGHTNESS) {
      issues.push('Too bright');
      return 0.2;
    }
    return 0;
  }

  private checkFaceSize(bbox: FaceBBox | null, frameWidth: number, frameHeight: number, issues: string[]): number {
    if (!bbox || frameWidth <= 0 || frameHeight <= 0) return 0;
    const ratio = (bbox.w * bbox.h) / (frameWidth * frameHeight);
    if (ratio < this.MIN_FACE_RATIO) {
      issues.push('Too far from camera');
      return 0.25;
    }
    if (ratio > this.MAX_FACE_RATIO) {
      issues.push('Too close to camera');
      return 0.15;
    }
    return 0;
  }

  private checkMotion(bbox: FaceBBox | null, issues: string[]): number {
    if (!bbox || !this.prevBBox) return 0;
    const dx = (bbox.x + bbox.w / 2) - (this.prevBBox.x + this.prevBBox.w / 2);
    const dy = (bbox.y + bbox.h / 2) - (this.prevBBox.y + this.prevBBox.h / 2);
    const motion = Math.hypot(dx, dy);
    this.motionRing[this.motionHead] = motion;
    this.motionHead = (this.motionHead + 1) % 30;
    if (this.motionCount < 30) this.motionCount++;

    let motionSum = 0;
    for (let k = 0; k < 30; k++) motionSum += this.motionRing[k];
    const avgMotion = motionSum / this.motionCount;
    if (avgMotion > this.MAX_MOTION_PX) {
      issues.push('Too much movement');
      return 0.3;
    }
    if (avgMotion > this.MAX_MOTION_PX * 0.6) {
      issues.push('Slight movement');
      return 0.1;
    }
    return 0;
  }

  private checkFPS(issues: string[]): number {
    const fps = this.getCurrentFPS();
    if (fps > 0 && fps < this.MIN_FPS) {
      issues.push(`Low FPS (${fps.toFixed(0)})`);
      return 0.2;
    }
    return 0;
  }

  /** Compute mean brightness (luminance) from RGBA ImageData — samples every 4th pixel. */
  private meanBrightness(imageData: ImageData): number {
    const { data } = imageData;
    let sum = 0;
    let count = 0;
    // Step by 16 bytes = 4 pixels, giving 1,296 samples instead of 5,184 (75% less work).
    for (let i = 0; i < data.length; i += 16) {
      sum += 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
      count++;
    }
    return sum / count;
  }

  /** Get current FPS from recent frame timestamps (ring-buffer variant). */
  getCurrentFPS(): number {
    if (this.tsCount < 2) return 0;
    const oldestIdx = this.tsCount >= 60 ? this.tsHead : 0;
    const newestIdx = (this.tsHead - 1 + 60) % 60;
    const elapsed = this.tsRing[newestIdx] - this.tsRing[oldestIdx];
    if (elapsed <= 0) return 0;
    return ((this.tsCount - 1) * 1000) / elapsed;
  }

  reset(): void {
    this.prevBBox = null;
    this.motionHead = 0;
    this.motionCount = 0;
    this.motionRing.fill(0);
    this.framesWithoutFace = 0;
    this.tsHead = 0;
    this.tsCount = 0;
    this.tsRing.fill(0);
  }
}
