import type { GenerationStep } from '../lib/generate'

interface GenerationStageProps {
  prompt: string
  generatedText: string
  currentStep: GenerationStep | null
  /** Whether the sampled token should be highlighted yet (two-phase reveal per step). */
  revealed: boolean
  stepCount: number
  maxSteps: number
}

export function GenerationStage({
  prompt,
  generatedText,
  currentStep,
  revealed,
  stepCount,
  maxSteps,
}: GenerationStageProps) {
  return (
    <section className="stage stage-generating">
      <p className="progress-text">
        {stepCount} / {maxSteps} トークン
      </p>

      <p className="growing-text">
        {prompt}
        <span className="growing-text-new">{generatedText}</span>
        <span className="cursor-blink">|</span>
      </p>

      {currentStep && (
        <div className="candidate-list">
          {currentStep.candidates.map((c) => {
            const isChosen = revealed && c.id === currentStep.chosen.id
            return (
              <div key={c.id} className={`candidate-row${isChosen ? ' candidate-row-chosen' : ''}`}>
                <span className="candidate-piece">{c.piece || '∅'}</span>
                <span className="candidate-bar-track">
                  <span className="candidate-bar-fill" style={{ width: `${Math.min(c.prob * 100, 100)}%` }} />
                </span>
                <span className="candidate-prob">{(c.prob * 100).toFixed(1)}%</span>
              </div>
            )
          })}
        </div>
      )}
    </section>
  )
}
