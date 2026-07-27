import { defineConfig } from 'vite';
import solidPlugin from 'vite-plugin-solid';
// import devtools from 'solid-devtools/vite';

export default defineConfig({
  plugins: [
    /* 
    Uncomment the following line to enable solid-devtools.
    For more info see https://github.com/thetarnav/solid-devtools/tree/main/packages/extension#readme
    */
    // devtools(),
    solidPlugin(),
  ],
  server: {
    port: 3000,
  },
  build: {
    target: 'esnext',
    rollupOptions: {
      // The diagnostics pages ship too: selftest.html is the only thing that
      // checks the WGSL arithmetic against the BigInt oracle on real hardware.
      input: {
        main: 'index.html',
        selftest: 'selftest.html',
        gpu: 'gpu.html',
      },
    },
  },
});
