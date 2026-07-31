import {
  LogitsProcessorList,
  NoRepeatNGramLogitsProcessor,
  RepetitionPenaltyLogitsProcessor,
  type PreTrainedModel,
  type PreTrainedTokenizer,
} from '@huggingface/transformers'

// Small models like this one degrade into repeating the same token/phrase far more readily than
// larger ones (see the <unk>-loop note below). These defaults lean fairly strong to counter that;
// 1.0 disables the penalty, 0 disables the ngram ban. See `startGeneration`'s `options` param.
const DEFAULT_REPETITION_PENALTY = 1.3
const DEFAULT_NO_REPEAT_NGRAM_SIZE = 3

export interface TokenCandidate {
  id: number
  piece: string
  prob: number
}

export interface GenerationStep {
  /** Top candidate next tokens, sorted by probability descending. */
  candidates: TokenCandidate[]
  /** The token actually appended to the sequence — sampled, unless `forced` is true. */
  chosen: TokenCandidate
  isEos: boolean
  /** True if `chosen` was injected via `forcedId` rather than sampled from the model's distribution. */
  forced: boolean
}

// This tokenizer's special tokens occupy ids 0-6: <unk>, <s>, </s>, [PAD], [CLS], [SEP], [MASK]
// (verified by decoding ids 0-9 directly; there's no per-id metadata to read this from).
const EOS_TOKEN_ID = 2
const ALL_SPECIAL_TOKEN_IDS = new Set([0, 1, 2, 3, 4, 5, 6])
// All but </s> (2) are excluded from actual sampling — they're shown in the candidate list
// for transparency (e.g. "the model wasn't confident, <unk> ranked high"), but would just be
// confusing noise if they ended up in the generated sentence itself. </s> is kept selectable
// since choosing it has real meaning: it ends generation (subject to the `allowEos` step option).
const NON_EOS_SPECIAL_TOKEN_IDS = new Set([...ALL_SPECIAL_TOKEN_IDS].filter((id) => id !== EOS_TOKEN_ID))

// Sentencepiece word-boundary/meta pieces sometimes decode to an empty string in isolation.
// Selecting one wouldn't visibly change the generated text at all (see the growing-text UI in
// scene 2), and in Japanese text (no inter-word spacing) they add nothing — so they're excluded
// from sampling the same way special tokens are, while still shown (as "∅") in the candidate list.
function isSelectablePiece(id: number, piece: string): boolean {
  return !NON_EOS_SPECIAL_TOKEN_IDS.has(id) && piece !== ''
}

// How many top-probability ids to decode+inspect when building the selectable pool. Must be
// comfortably larger than `topK` since some of the top ids get filtered out by isSelectablePiece.
const SELECTION_POOL_SIZE = 50

export interface GenerationSession {
  model: PreTrainedModel
  tokenizer: PreTrainedTokenizer
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  generationConfig: any
  allInputIds: bigint[][]
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  modelInputs: Record<string, any>
  promptLength: number
  logitsProcessor: LogitsProcessorList
}

export interface GenerationOptions {
  /** 1.0 = no penalty; higher discourages re-picking already-used tokens. */
  repetitionPenalty?: number
  /** Bans repeating any token n-gram of this size that's already occurred; 0 disables. */
  noRepeatNgramSize?: number
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
  options: GenerationOptions = {},
): Promise<GenerationSession> {
  const { repetitionPenalty = DEFAULT_REPETITION_PENALTY, noRepeatNgramSize = DEFAULT_NO_REPEAT_NGRAM_SIZE } = options
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const encoded = await (tokenizer as any)(prompt)
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const generationConfig = (model as any)._prepare_generation_config(null, {})

  const logitsProcessor = new LogitsProcessorList()
  if (repetitionPenalty !== 1.0) logitsProcessor.push(new RepetitionPenaltyLogitsProcessor(repetitionPenalty))
  if (noRepeatNgramSize > 0) logitsProcessor.push(new NoRepeatNGramLogitsProcessor(noRepeatNgramSize))

  return {
    model,
    tokenizer,
    generationConfig,
    allInputIds: encoded.input_ids.tolist(),
    modelInputs: { input_ids: encoded.input_ids, attention_mask: encoded.attention_mask },
    promptLength: encoded.input_ids.dims[1],
    logitsProcessor,
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
  /**
   * If set, skip sampling and append this token id instead — used to guarantee a user-picked
   * word actually appears in the generated text (see `planForcedWords`). Candidates are still
   * computed/returned as normal so the UI can show what the model would have picked on its own.
   */
  forcedId?: number
  /** Whether </s> may be sampled this step. Set false while there are still forced words pending,
   * so the model can't end the sentence before honoring the "文章を作ります" promise. Ignored when
   * `forcedId` is set. Default true. */
  allowEos?: boolean
}

function isEosId(generationConfig: { eos_token_id: number | number[] }, id: number): boolean {
  const eosTokenId = generationConfig.eos_token_id
  return Array.isArray(eosTokenId) ? eosTokenId.includes(id) : id === eosTokenId
}

/**
 * Runs one generation step: forward pass -> next-token probability distribution ->
 * pick top-K candidates for display -> sample (or force) one to append to the sequence.
 *
 * Greedy (always picking the top candidate) tends to degenerate into `<unk>`-token loops
 * on this small model, so this always samples rather than taking argmax — see STEPS.md.
 */
export async function stepGeneration(session: GenerationSession, options: StepOptions = {}): Promise<GenerationStep> {
  const { topK = 10, temperature = 0.8, forcedId, allowEos = true } = options
  const { model, tokenizer, generationConfig } = session

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  session.modelInputs = (model as any).prepare_inputs_for_generation(
    session.allInputIds,
    session.modelInputs,
    generationConfig,
  )
  const outputs = await model.forward(session.modelInputs)
  // Penalize/ban already-used tokens (repetition_penalty, no_repeat_ngram_size) before softmax,
  // so both the displayed candidates and the selection pool reflect the adjusted distribution —
  // this is a real steer on the model's output, not just a display-time filter like isSelectablePiece.
  const lastLogitsTensor = outputs.logits.slice(null, -1, null)
  session.logitsProcessor._call(session.allInputIds, lastLogitsTensor)
  const lastLogits = lastLogitsTensor.data as Float32Array
  const probs = softmax(lastLogits)

  const sortedIndices = [...probs.keys()].sort((a, b) => probs[b] - probs[a])
  const poolIndices = sortedIndices.slice(0, Math.max(topK, SELECTION_POOL_SIZE))
  const pool: TokenCandidate[] = poolIndices.map((id) => ({
    id,
    piece: tokenizer.decode([id]),
    prob: probs[id],
  }))

  // Candidates shown in the UI: raw top-K, unfiltered (includes special/empty-piece tokens for
  // transparency — e.g. "the model wasn't confident, <unk> ranked high").
  const candidates = pool.slice(0, topK)

  const forced = forcedId !== undefined
  let chosen: TokenCandidate
  if (forced) {
    chosen = { id: forcedId, piece: tokenizer.decode([forcedId]), prob: probs[forcedId] ?? 0 }
  } else {
    // Selection pool: filtered (see isSelectablePiece) so the generated text itself never contains
    // raw "<unk>"/"[PAD]"/etc, or an invisible token that doesn't visibly grow the on-screen text.
    // Also drops </s> when allowEos is false (still pending forced words to weave in).
    const selectable = pool
      .filter((c) => isSelectablePiece(c.id, c.piece) && (allowEos || !isEosId(generationConfig, c.id)))
      .slice(0, topK)
    chosen = sampleFrom(selectable, temperature)
  }

  session.allInputIds[0].push(BigInt(chosen.id))
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  session.modelInputs = (model as any)._update_model_kwargs_for_generation({
    generated_input_ids: [[BigInt(chosen.id)]],
    outputs,
    model_inputs: session.modelInputs,
    is_encoder_decoder: false,
  })

  return { candidates, chosen, isEos: isEosId(generationConfig, chosen.id), forced }
}

/**
 * Encodes `word` and strips the tokenizer's usual leading boundary-marker/trailing-`</s>`
 * artifacts (`encode()` always appends an eos id, even for a single standalone word — unlike
 * `isSelectablePiece`, this must drop it too, not just the non-eos specials), leaving just the
 * "real" content token id(s) for the word on its own. Used to force a user-picked word into the
 * generated text — see `stepGeneration`'s `forcedId`.
 */
export function tokenizeWord(tokenizer: PreTrainedTokenizer, word: string): number[] {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const ids = (tokenizer as any).encode(word) as number[]
  return ids.filter((id) => !ALL_SPECIAL_TOKEN_IDS.has(id) && tokenizer.decode([id]) !== '')
}

export function decodeGenerated(session: GenerationSession): string {
  const generatedIds = session.allInputIds[0].slice(session.promptLength).map(Number)
  return session.tokenizer.decode(generatedIds)
}
