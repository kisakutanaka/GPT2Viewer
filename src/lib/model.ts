import { AutoModelForCausalLM, AutoTokenizer, env } from '@huggingface/transformers'

export interface ModelDef {
  /** Path-safe key, also the folder name under public/models/. */
  id: string
  /** Human-readable label for the UI. */
  label: string
  /** Original Hugging Face source repo, for crediting. */
  sourceRepo: string
}

// rinna/japanese-gpt2-xsmall (MIT license), re-exported to ONNX ourselves with
// standard dynamic int8 quantization so it loads reliably in onnxruntime-web (wasm).
// See STEPS.md Step 1 notes for why the hub's pre-converted ONNX build doesn't work.
export const BASE_MODEL: ModelDef = {
  id: 'rinna-japanese-gpt2-xsmall',
  label: 'rinna GPT-2 (xsmall)',
  sourceRepo: 'rinna/japanese-gpt2-xsmall',
}

env.allowLocalModels = true
env.allowRemoteModels = false
env.localModelPath = `${import.meta.env.BASE_URL}models/`

export async function loadModel(def: ModelDef = BASE_MODEL) {
  const tokenizer = await AutoTokenizer.from_pretrained(def.id)
  const model = await AutoModelForCausalLM.from_pretrained(def.id, {
    dtype: 'q8',
  })
  return { tokenizer, model }
}
