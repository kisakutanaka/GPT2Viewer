import { useRef, useState } from 'react'
import { loadModel, MODELS, DOUYOU_MODEL, AOZORA_MODEL } from './lib/model'
import { startGeneration, stepGeneration, decodeGenerated, tokenizeWord, type GenerationStep } from './lib/generate'
import type { PreTrainedModel, PreTrainedTokenizer } from '@huggingface/transformers'
import { WordSelectStage, type StyleOption } from './components/WordSelectStage'
import { GenerationStage } from './components/GenerationStage'
import { ResultStage } from './components/ResultStage'
import './App.css'

const WORD_CHOICES = [
  '猫', '今日', '学校', '音楽', '旅行', '未来', '友達', '料理', '宇宙', '花', '海', '本',
  '犬', '星', '夢', '家族', '雨', '山', '電車', '夏', '桜', '光', '時間', '冒険',
]
const WORDS_TO_PICK = 3
const MAX_NEW_TOKENS = 80
// Pacing for scene 2, split into two phases per token so the "候補と確率が表示される → 一つ選ばれる"
// moment is easy to follow rather than flashing by. Tune these two independently:
// - CANDIDATE_DISPLAY_MS: candidates shown with bars, nothing highlighted yet (時間をかけて確率を見る)
// - CHOSEN_HOLD_MS: the sampled token is highlighted, held before it's appended to the text
const CANDIDATE_DISPLAY_MS = 750
const CHOSEN_HOLD_MS = 500

// 名言 still uses the base model — its fine-tuned corpus hasn't been sourced yet (see
// STEPS.md Step 10). 詩/小説 use their respective fine-tuned models. Swap the remaining
// `modelId` once the last fine-tune exists.
const STYLE_OPTIONS: (StyleOption & { modelId: string })[] = [
  { key: 'novel', label: '小説', modelId: AOZORA_MODEL.id },
  { key: 'quote', label: '名言', modelId: MODELS[0].id },
  { key: 'poem', label: '詩', modelId: DOUYOU_MODEL.id },
]

type Scene = 'select' | 'generating' | 'result'

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

// Sentencepiece often fuses trailing punctuation onto the previous subword (e.g. "ので、" or
// "した。" as one token), so an exact match against "、"/"。" alone misses most real boundaries.
// Checking suffixes instead — plus common sentence-final endings — catches those fused tokens too.
const BOUNDARY_SUFFIXES = [
  // punctuation / closing marks (both full- and half-width where relevant)
  '、', '。', '！', '!', '？', '?', '…', '・', '」', '』', '）', ')', '】', ']',
  // common sentence-final verb/copula endings
  'です', 'ます', 'でした', 'ました', 'ません', 'でしょう',
]
function isBoundaryPiece(piece: string): boolean {
  return BOUNDARY_SUFFIXES.some((suffix) => piece.endsWith(suffix))
}

function App() {
  const [scene, setScene] = useState<Scene>('select')
  const [status, setStatus] = useState('')
  const [isLoadingModel, setIsLoadingModel] = useState(false)
  const [selectedWords, setSelectedWords] = useState<string[]>([])
  const [selectedStyle, setSelectedStyle] = useState<string | null>(null)
  const [introText, setIntroText] = useState('')
  const [storyPrefix, setStoryPrefix] = useState('')
  const [generatedText, setGeneratedText] = useState('')
  const [currentStep, setCurrentStep] = useState<GenerationStep | null>(null)
  const [revealed, setRevealed] = useState(false)
  const [stepCount, setStepCount] = useState(0)

  const modelCacheRef = useRef(new Map<string, { model: PreTrainedModel; tokenizer: PreTrainedTokenizer }>())

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

    setStatus('モデルを読み込み中...')
    setIsLoadingModel(true)
    let loaded: { model: PreTrainedModel; tokenizer: PreTrainedTokenizer }
    try {
      loaded = await ensureModel(style.modelId)
    } catch (err) {
      console.error('model load failed:', err)
      setStatus(`読み込み失敗: ${String(err)}`)
      setIsLoadingModel(false)
      return
    }
    setIsLoadingModel(false)
    setStatus('')

    const [w1, w2, w3] = selectedWords
    const intro = `${w1}、${w2}、${w3}を使った文章を作ります。`
    const story = `${w1}`
    setIntroText(intro)
    setStoryPrefix(story)
    setGeneratedText('')
    setCurrentStep(null)
    setStepCount(0)
    setScene('generating')

    try {
      const { model, tokenizer } = loaded
      // Only `story` (the seed word) is fed to the model — `intro` is narration text shown in
      // the UI (see introText below) but never appeared in any fine-tuning corpus, and feeding
      // it as context measurably dilutes a fine-tuned style back toward generic/base-model-like
      // output (confirmed by comparing generation with vs without it as a prompt prefix).
      const session = await startGeneration(model, tokenizer, story)

      // w1 is already baked into `story`. To keep the "w1、w2、w3を使った文章を作ります" promise,
      // w2/w3 aren't left to chance — their tokens get force-inserted (stepGeneration's forcedId)
      // roughly around the 30%/60% marks, but only once the model has just produced a natural
      // clause boundary (読点/句点), so the insertion doesn't land mid-word/mid-clause. If no
      // boundary shows up within FORCE_MAX_EXTRA_WAIT tokens past the checkpoint, force anyway —
      // the guarantee that all 3 words appear matters more than always finding a perfect spot.
      // </s> is disallowed for the entire run — generation always uses the full MAX_NEW_TOKENS
      // budget rather than possibly stopping early.
      const CHECKPOINT_FRACTIONS = [0.3, 0.6]
      const FORCE_MAX_EXTRA_WAIT = 8

      const pendingWords = [
        { word: w2, ids: tokenizeWord(tokenizer, w2) },
        { word: w3, ids: tokenizeWord(tokenizer, w3) },
      ]
      let pendingIndex = 0
      let forcedQueue: number[] = []

      for (let i = 0; i < MAX_NEW_TOKENS; i++) {
        const forcedId = forcedQueue.length > 0 ? forcedQueue.shift() : undefined

        const result = await stepGeneration(session, { forcedId, allowEos: false })
        setCurrentStep(result)
        setRevealed(false)
        setStepCount(i + 1)
        await sleep(CANDIDATE_DISPLAY_MS)

        setRevealed(true)
        await sleep(CHOSEN_HOLD_MS)

        const textSoFar = decodeGenerated(session)
        setGeneratedText(textSoFar)

        // On a free (non-forced) step, decide whether it's time to start forcing the next word.
        if (forcedId === undefined && pendingIndex < pendingWords.length) {
          const pending = pendingWords[pendingIndex]
          if (textSoFar.includes(pending.word)) {
            // The model already wrote this word on its own — forcing it in too would duplicate it.
            pendingIndex++
          } else {
            const checkpointStep = Math.round(MAX_NEW_TOKENS * CHECKPOINT_FRACTIONS[pendingIndex])
            const stepsPastCheckpoint = i + 1 - checkpointStep
            const atBoundary = isBoundaryPiece(result.chosen.piece)
            if (stepsPastCheckpoint >= 0 && (atBoundary || stepsPastCheckpoint >= FORCE_MAX_EXTRA_WAIT)) {
              forcedQueue = [...pending.ids]
              pendingIndex++
            }
          }
        }
      }
    } catch (err) {
      console.error('generation failed:', err)
      setStatus(`生成エラー: ${String(err)}`)
    }

    setScene('result')
  }

  function handleRestart() {
    // Free the ONNX session(s) for whichever model(s) got loaded this round, rather than
    // leaving them cached indefinitely — a kiosk-style usage pattern (many visitors, page
    // never reloaded) could otherwise accumulate every style's model in WASM memory over time,
    // which matters given the iPhone/mobile Safari memory constraint this app targets.
    const cachedModels = [...modelCacheRef.current.values()].map((c) => c.model)
    modelCacheRef.current.clear()
    for (const model of cachedModels) {
      model.dispose().catch((err) => console.error('model dispose failed:', err))
    }

    setSelectedWords([])
    setSelectedStyle(null)
    setIntroText('')
    setStoryPrefix('')
    setGeneratedText('')
    setCurrentStep(null)
    setStepCount(0)
    setScene('select')
  }

  if (scene === 'generating') {
    return (
      <GenerationStage
        introText={introText}
        storyPrefix={storyPrefix}
        generatedText={generatedText}
        currentStep={currentStep}
        revealed={revealed}
        stepCount={stepCount}
        maxSteps={MAX_NEW_TOKENS}
      />
    )
  }

  if (scene === 'result') {
    return (
      <ResultStage
        introText={introText}
        storyPrefix={storyPrefix}
        generatedText={generatedText}
        onRestart={handleRestart}
      />
    )
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
      isLoadingModel={isLoadingModel}
      status={status}
      onStart={handleStart}
    />
  )
}

export default App
