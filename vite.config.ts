import { defineConfig, type Plugin } from 'vite'
import { readFileSync, existsSync } from 'node:fs'
import { join } from 'node:path'

/**
 * Vite plugin that serves ONNX Runtime worker .mjs files from node_modules
 * during development. These files cannot live in public/ because Vite blocks
 * ES module imports from there (public/ assets are static-only).
 *
 * In production builds, Vite handles them as normal bundled assets.
 */
function serveOrtWasm(): Plugin {
  return {
    name: 'serve-onnxruntime-wasm',
    configureServer(server) {
      server.middlewares.use((req, res, next) => {
        const url = (req.url ?? '').split('?')[0];
        const match = url.match(/(ort-wasm[^/]*\.mjs)$/);
        if (match) {
          const file = join(process.cwd(), 'node_modules', 'onnxruntime-web', 'dist', match[1]);
          if (existsSync(file)) {
            res.setHeader('Content-Type', 'application/javascript');
            res.end(readFileSync(file, 'utf-8'));
            return;
          }
        }
        next();
      });
    },
  };
}

// https://vitejs.dev/config/
export default defineConfig({
  // Set to the repo name so all assets resolve correctly on GitHub Pages
  // e.g. https://asheesh1996.github.io/NeuroWASM/
  base: '/NeuroWASM/',

  plugins: [serveOrtWasm()],

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
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
})
