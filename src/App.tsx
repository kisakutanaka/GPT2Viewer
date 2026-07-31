import { useEffect, useState } from 'react'
import { loadModel } from './lib/model'
import './App.css'

const SAMPLE_TEXT = '今日はいい天気ですね'

function App() {
  const [status, setStatus] = useState('モデルを読み込み中...')

  useEffect(() => {
    let cancelled = false
    loadModel()
      .then(({ tokenizer, model }) => {
        if (cancelled) return
        console.log('tokenizer loaded:', tokenizer)
        console.log('model loaded:', model)
        setStatus(`モデル読み込み完了: ${model.constructor.name}`)

        const ids = tokenizer.encode(SAMPLE_TEXT)
        const decoded = tokenizer.decode(ids)
        const pieces = ids.map((id) => tokenizer.decode([id]))
        console.log('sample text:', SAMPLE_TEXT)
        console.log('token ids:', ids)
        console.log('decoded (roundtrip):', decoded)
        console.log(
          'token breakdown:',
          ids.map((id, i) => `${id}:"${pieces[i]}"`),
        )
      })
      .catch((err) => {
        console.error('model load failed:', err)
        if (!cancelled) setStatus(`読み込み失敗: ${String(err)}`)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return (
    <section id="center">
      <h1>Step 1-2: モデル & トークナイザー確認</h1>
      <p>{status}</p>
      <p>詳細はブラウザのコンソールを確認してください。</p>
    </section>
  )
}

export default App
