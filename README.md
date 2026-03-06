# NeuroWASM

**Real-time object detection in the browser — no server, no cloud, no latency.**

NeuroWASM runs YOLO models entirely on-device using [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html). It captures live webcam video, passes each frame through a YOLO ONNX model, and draws bounding boxes on a canvas overlay — all inside a single browser tab.

Two inference backends are supported:

| Backend | When to use |
|---|---|
| **WebGPU** | Modern Chromium — runs the model on your GPU via the WebGPU API |
| **CPU / WASM** | Any browser — runs the model in a WASM thread pool using SIMD + threads |

---

## Demo

```
Camera feed → preprocess → ONNX model → postprocess (NMS) → canvas bounding boxes
```

Detects all 80 COCO classes (person, car, dog, chair, …) with per-class colour coding and confidence scores.

---

## Getting Started

### Prerequisites

- [Node.js](https://nodejs.org/) 18+
- A Chromium-based browser (Chrome / Edge) for WebGPU support
- A YOLO ONNX model file (e.g. `yolo26n.onnx`) placed in the `public/` directory

### Install & run

```bash
git clone https://github.com/your-username/NeuroWASM.git
cd NeuroWASM
npm install
npm run dev
```

Open `http://localhost:5173` in Chrome. Grant camera permission when prompted.

### Build for production

```bash
npm run build      # outputs to dist/
npm run preview    # serve the built bundle locally
```

---

## Architecture

```
NeuroWASM/
├── index.html          # Single-page shell: video, canvas, Bootstrap controls
├── public/
│   └── yolo26n.onnx    # YOLO model weights (add your own here)
└── src/
    ├── main.ts         # App entry point & frame loop
    ├── modelManager.ts # ONNX session lifecycle + pre/postprocessing
    ├── camera.ts       # MediaDevices camera abstraction
    ├── ui.ts           # Canvas renderer + HTML control wiring
    └── style.css       # Minimal global styles
```

### Data flow per frame

```
┌─────────────────────────────────────────────────────────────┐
│  main.ts — serialised async loop (one frame at a time)      │
│                                                             │
│  requestAnimationFrame ──► camera.getVideoElement()         │
│                              │                              │
│                       modelManager.runInference()           │
│                              │                              │
│                     ┌────────▼────────┐                     │
│                     │  preprocess()   │                     │
│                     │  letterbox pad  │                     │
│                     │  → Float32[640] │                     │
│                     └────────┬────────┘                     │
│                              │                              │
│                     ┌────────▼────────┐                     │
│                     │  session.run()  │                     │
│                     │  WebGPU / WASM  │                     │
│                     └────────┬────────┘                     │
│                              │                              │
│                     ┌────────▼────────┐                     │
│                     │  postprocess()  │                     │
│                     │  NMS filtering  │                     │
│                     │  coord rescale  │                     │
│                     └────────┬────────┘                     │
│                              │                              │
│                        ui.drawBoxes() ──► <canvas>          │
└─────────────────────────────────────────────────────────────┘
```

### Key design decisions

**Serialised frame loop** — inference is `await`-ed before the next `requestAnimationFrame` is requested. This prevents WebGPU buffer race conditions (`Buffer was unmapped before mapping was resolved`) that occur when the GPU is still finishing a previous `OrtRun` when the next frame fires.

**Letterbox preprocessing** — frames are resized to 640×640 with padding (not stretching) and the padding offsets are tracked so bounding boxes are correctly projected back to native video coordinates.

**Hot-swap backends** — changing the backend dropdown disposes the current ONNX session (`session.release()`) and creates a new one without a page reload. The frame loop exits cleanly via an `isRunning` flag.

**COCO 80-class support** — class names and per-class HSL colours are generated once at startup using golden-angle hue spacing for maximum visual contrast.

---

## GPU Performance

### Why WebGPU and not WebGL?

ONNX Runtime Web's WebGPU backend uses compute shaders, which map directly to the GPU's tensor math units. WebGL can only express GPU computation through rasterization APIs (vertex/fragment shaders), which is significantly slower for matrix operations.

### Forcing Chrome to use your NVIDIA GPU

On Windows laptops with hybrid graphics (Intel integrated + NVIDIA discrete), Chrome often defaults to the Intel GPU for WebGPU — even when the "High Performance" option is selected in the backend dropdown. This is a [known Chromium limitation](https://bugs.chromium.org/p/chromium/issues/detail?id=1307634): the `powerPreference` hint in `requestAdapter()` is currently ignored on Windows.

**To force Chrome to use your NVIDIA GPU:**

#### Option A — Windows Graphics Settings (recommended)

1. Open **Settings → System → Display**
2. Scroll down and click **Graphics settings**
3. Under "Add an app", choose **Desktop app** from the dropdown and click **Browse**
4. Navigate to your Chrome executable:
   - Usually `C:\Program Files\Google\Chrome\Application\chrome.exe`
5. Once Chrome appears in the list, click **Options**
6. Select **High performance** and click **Save**
7. **Restart Chrome completely** (close all windows, wait a few seconds, reopen)

#### Option B — NVIDIA Control Panel

1. Right-click the desktop → **NVIDIA Control Panel**
2. Go to **3D Settings → Manage 3D Settings → Program Settings**
3. In the dropdown, select **Google Chrome** (add it manually if not listed)
4. Set **"Preferred graphics processor"** to **High-performance NVIDIA processor**
5. Click **Apply** and restart Chrome

After either change, open Chrome DevTools console and look for:

```
🎮 WebGPU Adapter: nvidia — NVIDIA GeForce RTX ... (high-performance)
```

This confirms Chrome is now routing WebGPU to your NVIDIA card.

---

## Controls

| Control | Description |
|---|---|
| Camera dropdown | Switch between available webcam inputs |
| Model dropdown | Select the ONNX model file to load |
| Backend dropdown | Switch inference backend (reloads the model) |
| Status badge | Shows current state: Initializing / Running / Error |
| FPS badge | Inference throughput (frames per second) |

---

## Adding a Model

1. Export your YOLO model to ONNX format with NMS embedded (output shape `[1, N, 6]` — `[x1, y1, x2, y2, conf, classId]` in absolute 640×640 coordinates).
2. Place the `.onnx` file in `public/`.
3. Add a `<option>` entry for it in `index.html` in the `#model-select` dropdown.

Non-NMS formats (`[1, 4+80, anchors]` raw anchor outputs) are also supported — the postprocessor autodetects the layout and applies NMS internally.

---

## Tech Stack

| | |
|---|---|
| **Runtime** | [ONNX Runtime Web](https://onnxruntime.ai/) 1.24 |
| **Build tool** | [Vite](https://vitejs.dev/) 7 |
| **Language** | TypeScript 5.9 |
| **UI** | Bootstrap 5.3 + vanilla Canvas API |
| **GPU API** | WebGPU (via ORT WebGPU backend) |
| **CPU fallback** | WASM with SIMD + multi-threading |

---

## Browser Support

| Browser | WebGPU | WASM fallback |
|---|---|---|
| Chrome 113+ | ✅ | ✅ |
| Edge 113+ | ✅ | ✅ |
| Firefox | ⚠️ experimental flag | ✅ |
| Safari 18+ | ✅ (macOS/iOS) | ✅ |

WebGPU must be enabled. In older Chrome builds you can force-enable it at `chrome://flags/#enable-unsafe-webgpu`.

---

## License

MIT
