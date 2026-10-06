import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  build: {
    target: 'es2022',
    outDir: '../static/labeler',
    emptyOutDir: true,
    modulePreload: false,
    rollupOptions: {
      input: 'src/main.jsx',
      output: { entryFileNames: 'labeler.js', assetFileNames: 'labeler.[ext]' },
    },
  },
});
