import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  // GitHub Pages はリポジトリ名がサブパスになる（例: https://<user>.github.io/GPT2Viewer/）
  // 実際のリポジトリ名がフォルダ名と異なる場合はここを合わせて変更する
  base: '/GPT2Viewer/',
})
