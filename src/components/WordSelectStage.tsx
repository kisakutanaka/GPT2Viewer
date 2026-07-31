export interface StyleOption {
  key: string
  label: string
}

interface WordSelectStageProps {
  wordChoices: string[]
  wordsToPick: number
  selectedWords: string[]
  onToggleWord: (word: string) => void
  styleOptions: StyleOption[]
  selectedStyle: string | null
  onSelectStyle: (key: string) => void
  isLoadingModel: boolean
  status: string
  onStart: () => void
}

export function WordSelectStage({
  wordChoices,
  wordsToPick,
  selectedWords,
  onToggleWord,
  styleOptions,
  selectedStyle,
  onSelectStyle,
  isLoadingModel,
  status,
  onStart,
}: WordSelectStageProps) {
  const wordsReady = selectedWords.length === wordsToPick
  const canStart = wordsReady && selectedStyle !== null && !isLoadingModel

  return (
    <section className="stage stage-select">
      <h1 className="app-title">さくぶんAI</h1>

      <p className="stage-prompt">
        好きな単語を{wordsToPick}つ選んでね（{selectedWords.length}/{wordsToPick}）
      </p>
      <div className="word-grid">
        {wordChoices.map((word) => (
          <button
            key={word}
            className="word-button"
            onClick={() => onToggleWord(word)}
            disabled={!selectedWords.includes(word) && selectedWords.length >= wordsToPick}
            aria-pressed={selectedWords.includes(word)}
          >
            {word}
          </button>
        ))}
      </div>

      {wordsReady && (
        <>
          <p className="stage-prompt">どの文体で生成する？</p>
          <div className="style-grid">
            {styleOptions.map((style) => (
              <button
                key={style.key}
                className="style-button"
                onClick={() => onSelectStyle(style.key)}
                aria-pressed={selectedStyle === style.key}
              >
                {style.label}
              </button>
            ))}
          </div>
        </>
      )}

      <button className="primary-button" onClick={onStart} disabled={!canStart}>
        {isLoadingModel ? 'モデルを読み込み中...' : '作文する'}
      </button>

      {status && <p className="status-text">{status}</p>}
    </section>
  )
}
