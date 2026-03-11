// modelManager.ts
import * as ort from 'onnxruntime-web/webgpu';

export interface BoundingBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classIndex: number;
  label: string;
}

export type Backend = 'webgpu-high-performance' | 'webgpu-low-power' | 'wasm';

export class ModelManager {
  private session: ort.InferenceSession | null = null;
  private modelName: string;
  private backend: Backend;
  private isProcessing: boolean = false;
  private inputWidth: number = 640;
  private inputHeight: number = 640;
  private hasLoggedShape: boolean = false;
  private debugFrameCount: number = 0;

  // COCO 80 class names (standard YOLO classes)
  private classNames: string[] = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
  ];

  // Per-class colors for nicer visualization
  private classColors: string[];

  // Pre-allocated preprocessing resources — avoid DOM canvas creation and 4.7 MB
  // Float32Array allocation on every inference call.
  private readonly preprocessCanvas: OffscreenCanvas;
  private readonly preprocessCtx: OffscreenCanvasRenderingContext2D;
  private readonly preprocessBuffer: Float32Array;

  constructor(modelName: string, backend: Backend = 'webgpu-high-performance') {
    this.modelName = modelName;
    this.backend = backend;
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/';

    // Generate distinct colors for each class
    this.classColors = this.classNames.map((_, i) => {
      const hue = (i * 137.508) % 360; // golden angle spacing
      return `hsl(${hue}, 90%, 55%)`;
    });

    this.preprocessCanvas = new OffscreenCanvas(this.inputWidth, this.inputHeight);
    this.preprocessCtx = this.preprocessCanvas.getContext('2d', { willReadFrequently: true })!;
    this.preprocessBuffer = new Float32Array(3 * this.inputWidth * this.inputHeight);
  }

  async dispose(): Promise<void> {
    if (this.session) {
      await this.session.release();
      this.session = null;
    }
    this.hasLoggedShape = false;
    this.debugFrameCount = 0;
  }

  getClassColor(classIndex: number): string {
    return this.classColors[classIndex % this.classColors.length] || '#00ff00';
  }

  async initialize(): Promise<boolean> {
    try {
      // Use Vite's BASE_URL so the path is correct both locally ("/")
      // and on GitHub Pages ("/NeuroWASM/"). BASE_URL always ends with "/".
      const modelUrl: string = `${import.meta.env.BASE_URL}${this.modelName}`;

      if (this.backend === 'wasm') {
        const opt: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'] };
        this.session = await ort.InferenceSession.create(modelUrl, opt);
        console.log('%c✅ Model loaded — CPU (WASM)', 'color: #ffcc00; font-weight: bold');
      } else {
        const powerPreference: 'high-performance' | 'low-power' =
          this.backend === 'webgpu-high-performance' ? 'high-performance' : 'low-power';

        // Probe the adapter to log which GPU the browser chose
        if ('gpu' in navigator) {
          try {
            const adapter = await (navigator as any).gpu.requestAdapter({ powerPreference });
            if (adapter) {
              const info = await adapter.requestAdapterInfo();
              console.log(
                `%c🎮 WebGPU Adapter: ${info.vendor ?? '?'} — ${info.device ?? info.description ?? 'unknown'} (${powerPreference})`,
                'color: #00ff88; font-weight: bold'
              );
            }
          } catch { /* info not critical */ }
        }

        try {
          const opt: ort.InferenceSession.SessionOptions = {
            executionProviders: [{ name: 'webgpu', powerPreference } as any]
          };
          this.session = await ort.InferenceSession.create(modelUrl, opt);
          console.log('%c✅ Model loaded — WebGPU', 'color: #00ff88; font-weight: bold');
        } catch (e) {
          console.warn('WebGPU failed, falling back to WASM:', e);
          const opt: ort.InferenceSession.SessionOptions = { executionProviders: ['wasm'] };
          this.session = await ort.InferenceSession.create(modelUrl, opt);
          console.log('%c✅ Model loaded — WASM (WebGPU unavailable)', 'color: #ffcc00; font-weight: bold');
        }
      }

      console.log('📥 Input names:', this.session.inputNames);
      console.log('📤 Output names:', this.session.outputNames);
      return true;
    } catch (error) {
      console.error(`Failed to load model ${this.modelName}:`, error);
      return false;
    }
  }

  async runInference(videoElement: HTMLVideoElement): Promise<BoundingBox[]> {
    if (!this.session || this.isProcessing) return [];
    this.isProcessing = true;

    try {
      const { tensor, padX, padY, scale } = this.preprocess(videoElement);

      const feeds: Record<string, ort.Tensor> = {};
      feeds[this.session.inputNames[0]] = tensor;

      const outputData = await this.session.run(feeds);

      // Parse ALL outputs for debugging on first few frames
      if (!this.hasLoggedShape) {
        this.hasLoggedShape = true;
        console.group('%c🔍 YOLO Output Debug Info', 'color: #00aaff; font-weight: bold');
        for (const name of this.session.outputNames) {
          const t = outputData[name];
          const raw = Array.from(t.data as Float32Array);
          console.log(`Output "${name}": dims=${JSON.stringify(t.dims)}, type=${t.type}`);
          // Print first 3 detections
          if (t.dims.length === 3 && Number(t.dims[2]) === 6) {
            const stride = 6;
            for (let i = 0; i < Math.min(3, Number(t.dims[1])); i++) {
              const b = i * stride;
              console.log(`  det[${i}]: x1=${raw[b].toFixed(1)} y1=${raw[b+1].toFixed(1)} x2=${raw[b+2].toFixed(1)} y2=${raw[b+3].toFixed(1)} conf=${raw[b+4].toFixed(4)} cls=${raw[b+5].toFixed(0)}`);
            }
          } else {
            console.log(`  data[0..7]=${raw.slice(0, 8).map(v => v.toFixed(4)).join(', ')}`);
          }
        }
        console.groupEnd();
      }

      const outputTensor = outputData[this.session.outputNames[0]];
      const boxes = this.postprocess(
        outputTensor,
        videoElement.videoWidth,
        videoElement.videoHeight,
        padX,
        padY,
        scale
      );

      this.isProcessing = false;
      return boxes;
    } catch (error) {
      console.error('Inference error:', error);
      this.isProcessing = false;
      return [];
    }
  }

  private preprocess(videoElement: HTMLVideoElement): {
    tensor: ort.Tensor;
    padX: number;
    padY: number;
    scale: number;
  } {
    const ctx = this.preprocessCtx;

    // Letterbox: scale to fit inside inputWidth x inputHeight, centered with black padding
    const scale = Math.min(this.inputWidth / videoElement.videoWidth, this.inputHeight / videoElement.videoHeight);
    const drawW = Math.round(videoElement.videoWidth * scale);
    const drawH = Math.round(videoElement.videoHeight * scale);
    const padX = Math.round((this.inputWidth - drawW) / 2);
    const padY = Math.round((this.inputHeight - drawH) / 2);

    ctx.fillStyle = '#000000';
    ctx.fillRect(0, 0, this.inputWidth, this.inputHeight);
    ctx.drawImage(videoElement, padX, padY, drawW, drawH);

    const imageData = ctx.getImageData(0, 0, this.inputWidth, this.inputHeight);
    const { data } = imageData;
    const pixelCount = this.inputWidth * this.inputHeight;
    const float32Data = this.preprocessBuffer; // reuse pre-allocated buffer

    // NCHW format, normalized 0-1
    for (let i = 0; i < pixelCount; i++) {
      float32Data[i] = data[i * 4] / 255;                       // R
      float32Data[pixelCount + i] = data[i * 4 + 1] / 255;      // G
      float32Data[2 * pixelCount + i] = data[i * 4 + 2] / 255;  // B
    }

    return {
      tensor: new ort.Tensor('float32', float32Data, [1, 3, this.inputHeight, this.inputWidth]),
      padX,
      padY,
      scale
    };
  }

  private postprocess(
    output: ort.Tensor,
    videoWidth: number,
    videoHeight: number,
    padX: number,
    padY: number,
    scale: number
  ): BoundingBox[] {
    const dims = output.dims;
    const data = output.data as Float32Array;
    const confThreshold = 0.25;
    const validBoxes: BoundingBox[] = [];

    // ── Format A: [1, 4+numClasses, numAnchors]  (YOLOv8 transposed)
    // ── Format B: [1, numAnchors, 4+numClasses]  (row-major, sometimes YOLOv10/v26 e2e)
    // ── Format C: [1, numDetections, 6]           (NMS-embedded: x1,y1,x2,y2,conf,cls)
    // We auto-detect based on dims.

    if (dims.length === 3) {
      const d1 = Number(dims[1]);
      const d2 = Number(dims[2]);

      // CASE: NMS-free output [1, N, 6] — absolute xyxy coords + conf + classId
      if (d2 === 6 && d1 < 10000) {
        if (!this.hasLoggedShape) console.log('[Postprocess] Detected NMS-embedded format [1, N, 6]');
        for (let i = 0; i < d1; i++) {
          const base = i * 6;
          const x1m = data[base + 0];
          const y1m = data[base + 1];
          const x2m = data[base + 2];
          const y2m = data[base + 3];
          const conf = data[base + 4];
          const classIdx = Math.round(data[base + 5]);

          if (conf < confThreshold) continue;

          // Convert from model-space (640x640-letterboxed) to video coords
          validBoxes.push(this.mapBox(x1m, y1m, x2m, y2m, conf, classIdx, padX, padY, scale, videoWidth, videoHeight));
        }
        if (this.debugFrameCount < 5) {
          this.debugFrameCount++;
          console.log(`[Frame ${this.debugFrameCount}] ${validBoxes.length} detection(s) above conf=${confThreshold}`);
          validBoxes.slice(0, 3).forEach((b, i) => console.log(`  [${i}] ${b.label} ${(b.score*100).toFixed(1)}% @ (${b.x1.toFixed(0)},${b.y1.toFixed(0)})-(${b.x2.toFixed(0)},${b.y2.toFixed(0)}) vidSize=${videoWidth}x${videoHeight}`));
        }
        return validBoxes; // NMS already applied
      }

      // CASE: Format A — [1, 4+classes, anchors]  d1 < d2
      // CASE: Format B — [1, anchors, 4+classes]  d1 > d2
      // Heuristic: the larger dim is the anchor count
      let numAttributes: number;
      let numAnchors: number;
      let isFormatA: boolean;

      if (d1 <= d2) {
        // Format A: [1, attributes, anchors]
        numAttributes = d1;
        numAnchors = d2;
        isFormatA = true;
      } else {
        // Format B: [1, anchors, attributes]
        numAnchors = d1;
        numAttributes = d2;
        isFormatA = false;
      }

      const numClasses = numAttributes - 4;
      if (numClasses <= 0) {
        console.warn('[Postprocess] Unexpected output: numAttributes =', numAttributes);
        return [];
      }

      for (let i = 0; i < numAnchors; i++) {
        let cx: number, cy: number, w: number, h: number;

        if (isFormatA) {
          // data[attr * numAnchors + i]
          cx = data[0 * numAnchors + i];
          cy = data[1 * numAnchors + i];
          w  = data[2 * numAnchors + i];
          h  = data[3 * numAnchors + i];
        } else {
          // data[i * numAttributes + attr]
          cx = data[i * numAttributes + 0];
          cy = data[i * numAttributes + 1];
          w  = data[i * numAttributes + 2];
          h  = data[i * numAttributes + 3];
        }

        let maxScore = 0;
        let classIdx = -1;

        for (let j = 4; j < numAttributes; j++) {
          const score = isFormatA
            ? data[j * numAnchors + i]
            : data[i * numAttributes + j];

          if (score > maxScore) {
            maxScore = score;
            classIdx = j - 4;
          }
        }

        if (maxScore < confThreshold || classIdx < 0) continue;

        // YOLO outputs cx,cy,w,h in model-input space (640x640 letterboxed)
        const x1m = cx - w / 2;
        const y1m = cy - h / 2;
        const x2m = cx + w / 2;
        const y2m = cy + h / 2;

        validBoxes.push(this.mapBox(x1m, y1m, x2m, y2m, maxScore, classIdx, padX, padY, scale, videoWidth, videoHeight));
      }

      return this.applyNMS(validBoxes, 0.45);
    }

    // 2D output [numDetections, 6] — some exporter configs
    if (dims.length === 2) {
      const d0 = Number(dims[0]);  // number of detections
      const d1 = Number(dims[1]); // attributes per detection (≥6)
      if (d1 >= 6) {
        for (let i = 0; i < d0; i++) {
          const base = i * d1;
          const x1m = data[base + 0];
          const y1m = data[base + 1];
          const x2m = data[base + 2];
          const y2m = data[base + 3];
          const conf = data[base + 4];
          const classIdx = d1 === 6 ? Math.round(data[base + 5]) : this.argmax(data, base + 4, base + d1);

          if (conf < confThreshold) continue;
          validBoxes.push(this.mapBox(x1m, y1m, x2m, y2m, conf, classIdx, padX, padY, scale, videoWidth, videoHeight));
        }
        return validBoxes;
      }
    }

    console.warn('[Postprocess] Unrecognized output shape:', dims);
    return [];
  }

  /** Map from 640×640 letterboxed model-input space → original video coordinates */
  private mapBox(
    x1m: number, y1m: number, x2m: number, y2m: number,
    score: number, classIdx: number,
    padX: number, padY: number, scale: number,
    videoWidth: number, videoHeight: number
  ): BoundingBox {
    // Remove letterbox padding, then divide by scale
    let x1 = (x1m - padX) / scale;
    let y1 = (y1m - padY) / scale;
    let x2 = (x2m - padX) / scale;
    let y2 = (y2m - padY) / scale;

    // Clamp to video boundaries
    x1 = Math.max(0, Math.min(videoWidth, x1));
    y1 = Math.max(0, Math.min(videoHeight, y1));
    x2 = Math.max(0, Math.min(videoWidth, x2));
    y2 = Math.max(0, Math.min(videoHeight, y2));

    return {
      x1, y1, x2, y2,
      score,
      classIndex: classIdx,
      label: this.classNames[classIdx] ?? `class${classIdx}`
    };
  }

  private argmax(data: Float32Array, start: number, end: number): number {
    let maxVal = -Infinity;
    let maxIdx = 0;
    for (let i = start; i < end; i++) {
      if (data[i] > maxVal) { maxVal = data[i]; maxIdx = i - start; }
    }
    return maxIdx;
  }

  private applyNMS(boxes: BoundingBox[], iouThreshold: number): BoundingBox[] {
    boxes.sort((a, b) => b.score - a.score);
    const keep: BoundingBox[] = [];

    while (boxes.length > 0) {
      const best = boxes.shift()!;
      keep.push(best);

      boxes = boxes.filter(b => {
        // Only suppress if same class
        if (b.classIndex !== best.classIndex) return true;
        return this.iou(best, b) < iouThreshold;
      });
    }

    return keep;
  }

  private iou(a: BoundingBox, b: BoundingBox): number {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);

    const interW = Math.max(0, x2 - x1);
    const interH = Math.max(0, y2 - y1);
    const inter = interW * interH;
    if (inter === 0) return 0;

    const aArea = (a.x2 - a.x1) * (a.y2 - a.y1);
    const bArea = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (aArea + bArea - inter);
  }
}
