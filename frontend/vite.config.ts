/// <reference types="vitest/config" />
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import { fileURLToPath } from 'node:url';

// The Flask server serves the build at /ui/ and its assets from /static/ui/ (its CSP only allows
// same-origin scripts and styles, so everything is emitted as external files).
export default defineConfig({
  base: '/static/ui/',
  plugins: [react()],
  build: {
    outDir: fileURLToPath(new URL('../app/static/ui', import.meta.url)),
    emptyOutDir: true,
    sourcemap: false,
  },
  server: {
    port: 5173,
    // `npm run dev` talks to the real Flask API (python -m app.server, port 8010).
    // The proxy rewrites Host to the target (changeOrigin), but the browser's Origin stays
    // http://localhost:5173, which the server's CSRF guard rejects on POSTs. Present the proxied
    // request as same-origin to Flask; the guard itself stays strict.
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8010',
        changeOrigin: true,
        configure: (proxy) => {
          proxy.on('proxyReq', (proxyReq) => {
            if (proxyReq.getHeader('origin')) proxyReq.setHeader('origin', 'http://127.0.0.1:8010');
          });
        },
      },
    },
  },
  test: {
    environment: 'jsdom',
    globals: true,
    setupFiles: ['./src/test/setup.ts'],
    css: false,
  },
});
