import { AutoModelForCausalLM, AutoTokenizer, env } from '@huggingface/transformers'
import type { InferenceSession } from 'onnxruntime-common'

export interface ModelDef {
  /** Path-safe key, also the folder name under public/models/. */
  id: string
  /** Human-readable label for the UI. */
  label: string
  /** Original Hugging Face source repo, for crediting. */
  sourceRepo: string
  /** Extra ONNX Runtime session options this model's file needs, if any. */
  sessionOptions?: InferenceSession.SessionOptions
}

// rinna/japanese-gpt2-xsmall (MIT license), ONNX build from
// https://huggingface.co/saldra/rinna-japanese-gpt2-xsmall-onnx (also MIT), copied locally.
// Its quantized graph (int4 blockwise/MatMulNBits) only loads in onnxruntime-web if the
// 'TransposeDQWeightsForMatMulNBits' fusion optimization is skipped — see graphOptimizationLevel
// below and STEPS.md Step 1 notes for the full story (including why a self-converted int8
// alternative was tried and dropped: near-identical output quality, extra size, extra complexity).
export const BASE_MODEL: ModelDef = {
  id: 'rinna-japanese-gpt2-xsmall',
  label: 'rinna GPT-2 (xsmall)',
  sourceRepo: 'rinna/japanese-gpt2-xsmall',
  sessionOptions: { graphOptimizationLevel: 'basic' },
}

// The other 2 fine-tuned models required by Plan.md will be appended here once converted.
export const MODELS: ModelDef[] = [BASE_MODEL]

env.allowLocalModels = true
env.allowRemoteModels = false
env.localModelPath = `${import.meta.env.BASE_URL}models/`

export async function loadModel(def: ModelDef = BASE_MODEL) {
  const tokenizer = await AutoTokenizer.from_pretrained(def.id)
  const model = await AutoModelForCausalLM.from_pretrained(def.id, {
    dtype: 'q8',
    session_options: def.sessionOptions,
  })
  return { tokenizer, model }
}
