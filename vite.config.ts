import { defineConfig } from 'vite'

// https://vitejs.dev/config/
export default defineConfig({
  // Set to the repo name so all assets resolve correctly on GitHub Pages
  // e.g. https://asheesh1996.github.io/NeuroWASM/
  base: '/NeuroWASM/',

  build: {
    // Raise the chunk-size warning limit — ONNX runtime + WASM binaries are large
    chunkSizeWarningLimit: 20_000,

    rollupOptions: {
      output: {
        // Keep large vendor chunks separate so the browser can cache them
        manualChunks: {
          'onnxruntime': ['onnxruntime-web'],
        },
      },
    },
  },

  // Required so the browser can use SharedArrayBuffer (needed by ONNX WASM threads)
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
})
