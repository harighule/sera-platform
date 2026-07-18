import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
        secure: false,
        proxyTimeout: 30000,
        timeout: 30000,
        headers: {
          Connection: 'close'
        },
        configure: (proxy) => {
          proxy.on('error', (err) => {
            console.log('proxy error', err.message)
          })
        }
      },
      '/api/ws': {
        target: 'ws://localhost:8000',
        ws: true,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api\/ws/, '/ws')
      }
    }
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules')) {
            if (id.includes('react') || id.includes('react-dom') || id.includes('react-router-dom')) {
              return 'react-vendor';
            }
            if (id.includes('recharts')) {
              return 'chart-vendor';
            }
            if (id.includes('three')) {
              return 'three-vendor';
            }
          }
        }
      }
    }
  }
})
