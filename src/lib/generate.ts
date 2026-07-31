import type { PreTrainedModel, PreTrainedTokenizer } from '@huggingface/transformers'

export interface TokenCandidate {
  id: number
  piece: string
  prob: number
}

export interface GenerationStep {
  /** Top candidate next tokens, sorted by probability descending. */
  candidates: TokenCandidate[]
  /** The token actually sampled and appended to the sequence. */
  chosen: TokenCandidate
  isEos: boolean
}

export interface GenerationSession {
  model: PreTrainedModel
  tokenizer: PreTrainedTokenizer
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  generationConfig: any
  allInputIds: bigint[][]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelInputs: Record<string, any>
  promptLength: number
}

/**
 * NOTE: this relies on a few of transformers.js's internal-but-public helper methods
 * (`prepare_inputs_for_generation`, `_update_model_kwargs_for_generation`,
 * `_prepare_generation_config`) instead of the public `model.generate()`, because
 * `generate()` doesn't expose per-step logits/scores (see STEPS.md Step 3) — we need
 * those to show next-token candidates and probabilities as each token is produced.
 */
export async function startGeneration(
  model: PreTrainedModel,
  tokenizer: PreTrainedTokenizer,
  prompt: string,
): Promise<GenerationSession> {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const encoded = await (tokenizer as any)(prompt)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const generationConfig = (model as any)._prepare_generation_config(null, {})
  return {
    model,
    tokenizer,
    generationConfig,
    allInputIds: encoded.input_ids.tolist(),
    modelInputs: { input_ids: encoded.input_ids, attention_mask: encoded.attention_mask },
    promptLength: encoded.input_ids.dims[1],
  }
}

function softmax(logits: Float32Array | number[]): number[] {
  const max = Math.max(...logits)
  const exps = Array.from(logits, (x) => Math.exp(x - max))
  const sum = exps.reduce((a, b) => a + b, 0)
  return exps.map((x) => x / sum)
}

function sampleFrom(candidates: TokenCandidate[], temperature: number): TokenCandidate {
  const scaled = candidates.map((c) => Math.pow(c.prob, 1 / temperature))
  const sum = scaled.reduce((a, b) => a + b, 0)
  let r = Math.random() * sum
  for (let i = 0; i < candidates.length; i++) {
    r -= scaled[i]
    if (r <= 0) return candidates[i]
  }
  return candidates[candidates.length - 1]
}

export interface StepOptions {
  /** How many top candidates to return for display. */
  topK?: number
  /** Sampling temperature; lower = closer to greedy, higher = more random. */
  temperature?: number
}

/**
 * Runs one generation step: forward pass -> next-token probability distribution ->
 * pick top-K candidates for display -> sample one to actually append to the sequence.
 *
 * Greedy (always picking the top candidate) tends to degenerate into `<unk>`-token loops
 * on this small model, so this always samples rather than taking argmax — see STEPS.md.
 */
export async function stepGeneration(session: GenerationSession, options: StepOptions = {}): Promise<GenerationStep> {
  const { topK = 10, temperature = 0.8 } = options
  const { model, tokenizer, generationConfig } = session

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  session.modelInputs = (model as any).prepare_inputs_for_generation(
    session.allInputIds,
    session.modelInputs,
    generationConfig,
  )
  const outputs = await model.forward(session.modelInputs)
  const [, seqLen, vocabSize] = outputs.logits.dims
  const lastLogits = outputs.logits.data.slice((seqLen - 1) * vocabSize, seqLen * vocabSize)
  const probs = softmax(lastLogits)

  const topIndices = [...probs.keys()].sort((a, b) => probs[b] - probs[a]).slice(0, topK)
  const candidates: TokenCandidate[] = topIndices.map((id) => ({
    id,
    piece: tokenizer.decode([id]),
    prob: probs[id],
  }))

  const chosen = sampleFrom(candidates, temperature)
  session.allInputIds[0].push(BigInt(chosen.id))
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  session.modelInputs = (model as any)._update_model_kwargs_for_generation({
    generated_input_ids: [[BigInt(chosen.id)]],
    outputs,
    model_inputs: session.modelInputs,
    is_encoder_decoder: false,
  })

  const eosTokenId = generationConfig.eos_token_id
  const isEos = Array.isArray(eosTokenId) ? eosTokenId.includes(chosen.id) : chosen.id === eosTokenId

  return { candidates, chosen, isEos }
}

export function decodeGenerated(session: GenerationSession): string {
  const generatedIds = session.allInputIds[0].slice(session.promptLength).map(Number)
  return session.tokenizer.decode(generatedIds)
}
