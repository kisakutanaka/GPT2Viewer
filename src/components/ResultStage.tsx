interface ResultStageProps {
  introText: string
  storyPrefix: string
  generatedText: string
  onRestart: () => void
}

export function ResultStage({ introText, storyPrefix, generatedText, onRestart }: ResultStageProps) {
  return (
    <section className="stage stage-result">
      <h1 className="app-title">できあがり！</h1>
      <p className="intro-text">{introText}</p>
      <p className="result-text">
        {storyPrefix}
        {generatedText}
      </p>
      <button className="primary-button" onClick={onRestart}>
        もう一度体験する
      </button>
    </section>
  )
}
