interface ResultStageProps {
  prompt: string
  generatedText: string
  onRestart: () => void
}

export function ResultStage({ prompt, generatedText, onRestart }: ResultStageProps) {
  return (
    <section className="stage stage-result">
      <h1 className="app-title">できあがり！</h1>
      <p className="result-text">
        {prompt}
        {generatedText}
      </p>
      <button className="primary-button" onClick={onRestart}>
        もう一度体験する
      </button>
    </section>
  )
}
