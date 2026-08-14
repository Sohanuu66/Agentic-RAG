import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/ingest': 'http://localhost:8000',
      '/query':  'http://localhost:8000',
      '/eval':   'http://localhost:8000',
      '/health': 'http://localhost:8000',
    },
  },
})
