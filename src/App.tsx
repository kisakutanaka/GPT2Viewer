import { useEffect, useRef, useState } from 'react'
import { loadModel, MODELS } from './lib/model'
import { startGeneration, stepGeneration, decodeGenerated, type GenerationStep } from './lib/generate'
import type { PreTrainedModel, PreTrainedTokenizer } from '@huggingface/transformers'
import { WordSelectStage, type StyleOption } from './components/WordSelectStage'
import { GenerationStage } from './components/GenerationStage'
import { ResultStage } from './components/ResultStage'
import './App.css'

const WORD_CHOICES = ['猫', '今日', '学校', '音楽', '旅行', '未来', '友達', '料理', '宇宙', '花', '海', '本']
const WORDS_TO_PICK = 3
const MAX_NEW_TOKENS = 50
// Pacing for scene 2, split into two phases per token so the "候補と確率が表示される → 一つ選ばれる"
// moment is easy to follow rather than flashing by. Tune these two independently:
// - CANDIDATE_DISPLAY_MS: candidates shown with bars, nothing highlighted yet (時間をかけて確率を見る)
// - CHOSEN_HOLD_MS: the sampled token is highlighted, held before it's appended to the text
const CANDIDATE_DISPLAY_MS = 750
const CHOSEN_HOLD_MS = 750

// All 3 styles use the same base model for now (Plan.md's other 2 fine-tuned models
// haven't been sourced yet — see STEPS.md Step 7). Swap `modelId` per style once they exist.
const STYLE_OPTIONS: (StyleOption & { modelId: string })[] = [
  { key: 'novel', label: '小説', modelId: MODELS[0].id },
  { key: 'quote', label: '名言', modelId: MODELS[0].id },
  { key: 'poem', label: '詩', modelId: MODELS[0].id },
]

type Scene = 'select' | 'generating' | 'result'

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function App() {
  const [scene, setScene] = useState<Scene>('select')
  const [status, setStatus] = useState('モデルを読み込み中...')
  const [modelReady, setModelReady] = useState(false)
  const [selectedWords, setSelectedWords] = useState<string[]>([])
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null)
  const [prompt, setPrompt] = useState('')
  const [generatedText, setGeneratedText] = useState('')
  const [currentStep, setCurrentStep] = useState<GenerationStep | null>(null)
  const [revealed, setRevealed] = useState(false)
  const [stepCount, setStepCount] = useState(0)

  const modelCacheRef = useRef(new Map<string, { model: PreTrainedModel; tokenizer: PreTrainedTokenizer }>())

  useEffect(() => {
    let cancelled = false
    loadModel(MODELS[0])
      .then((loaded) => {
        if (cancelled) return
        modelCacheRef.current.set(MODELS[0].id, loaded)
        setStatus('準備OK')
        setModelReady(true)
      })
      .catch((err) => {
        console.error('model load failed:', err)
        if (!cancelled) setStatus(`読み込み失敗: ${String(err)}`)
      })
    return () => {
      cancelled = true
    }
  }, [])

  function toggleWord(word: string) {
    setSelectedWords((prev) => {
      if (prev.includes(word)) return prev.filter((w) => w !== word)
      if (prev.length >= WORDS_TO_PICK) return prev
      return [...prev, word]
    })
  }

  async function ensureModel(modelId: string) {
    const cached = modelCacheRef.current.get(modelId)
    if (cached) return cached
    const def = MODELS.find((m) => m.id === modelId) ?? MODELS[0]
    const loaded = await loadModel(def)
    modelCacheRef.current.set(modelId, loaded)
    return loaded
  }

  async function handleStart() {
    if (selectedWords.length !== WORDS_TO_PICK || !selectedStyle) return
    const style = STYLE_OPTIONS.find((s) => s.key === selectedStyle)
    if (!style) return

    const startPrompt = selectedWords.join('')
    setPrompt(startPrompt)
    setGeneratedText('')
    setCurrentStep(null)
    setStepCount(0)
    setScene('generating')

    try {
      const { model, tokenizer } = await ensureModel(style.modelId)
      const session = await startGeneration(model, tokenizer, startPrompt)

      for (let i = 0; i < MAX_NEW_TOKENS; i++) {
        const result = await stepGeneration(session)
        setCurrentStep(result)
        setRevealed(false)
        setStepCount(i + 1)
        await sleep(CANDIDATE_DISPLAY_MS)

        setRevealed(true)
        await sleep(CHOSEN_HOLD_MS)

        setGeneratedText(decodeGenerated(session))
        if (result.isEos) break
      }
    } catch (err) {
      console.error('generation failed:', err)
      setStatus(`生成エラー: ${String(err)}`)
    }

    setScene('result')
  }

  function handleRestart() {
    setSelectedWords([])
    setSelectedStyle(null)
    setPrompt('')
    setGeneratedText('')
    setCurrentStep(null)
    setStepCount(0)
    setScene('select')
  }

  if (scene === 'generating') {
    return (
      <GenerationStage
        prompt={prompt}
        generatedText={generatedText}
        currentStep={currentStep}
        revealed={revealed}
        stepCount={stepCount}
        maxSteps={MAX_NEW_TOKENS}
      />
    )
  }

  if (scene === 'result') {
    return <ResultStage prompt={prompt} generatedText={generatedText} onRestart={handleRestart} />
  }

  return (
    <WordSelectStage
      wordChoices={WORD_CHOICES}
      wordsToPick={WORDS_TO_PICK}
      selectedWords={selectedWords}
      onToggleWord={toggleWord}
      styleOptions={STYLE_OPTIONS}
      selectedStyle={selectedStyle}
      onSelectStyle={setSelectedStyle}
      modelReady={modelReady}
      status={status}
      onStart={handleStart}
    />
  )
}

export default App
