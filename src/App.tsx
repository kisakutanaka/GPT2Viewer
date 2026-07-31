import { useEffect, useState } from 'react'
import { loadModel } from './lib/model'
import { startGeneration, stepGeneration, decodeGenerated } from './lib/generate'
import './App.css'

const PROMPT = '今日は天気が良いので'
const MAX_NEW_TOKENS = 20

function App() {
  const [status, setStatus] = useState('モデルを読み込み中...')
  const [generatedText, setGeneratedText] = useState('')

  useEffect(() => {
    let cancelled = false

    async function run() {
      const { tokenizer, model } = await loadModel()
      if (cancelled) return
      console.log('tokenizer loaded:', tokenizer)
      console.log('model loaded:', model)
      setStatus(`モデル読み込み完了: ${model.constructor.name}`)

      const session = await startGeneration(model, tokenizer, PROMPT)
      console.log('generation start, prompt:', PROMPT)

      for (let i = 0; i < MAX_NEW_TOKENS; i++) {
        if (cancelled) return
        const { candidates, chosen, isEos } = await stepGeneration(session)
        console.log(
          `step ${i}: chose "${chosen.piece}" (${(chosen.prob * 100).toFixed(1)}%) | top candidates:`,
          candidates.map((c) => `${c.piece || '∅'}:${(c.prob * 100).toFixed(1)}%`).join(', '),
        )
        setGeneratedText(decodeGenerated(session))
        if (isEos) break
      }
    }

    run().catch((err) => {
      console.error('generation failed:', err)
      if (!cancelled) setStatus(`エラー: ${String(err)}`)
    })

    return () => {
      cancelled = true
    }
  }, [])

  return (
    <section id="center">
      <h1>Step 3-4: 次トークン予測 & 自動生成</h1>
      <p>{status}</p>
      <p>
        prompt: <strong>{PROMPT}</strong>
      </p>
      <p>
        generated: <strong>{generatedText}</strong>
      </p>
      <p>各ステップの候補トークンと確率はブラウザのコンソールを確認してください。</p>
    </section>
  )
}

export default App
