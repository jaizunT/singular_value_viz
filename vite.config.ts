import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  // REPLACE 'singular_value_viz' WITH YOUR ACTUAL GITHUB REPO NAME
  base: '/singular_value_viz/', 
})