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
// Its quantized graph (QDQ-format int8 — see quantize_config.json, not int4 as first assumed)
// only loads in onnxruntime-web if the 'TransposeDQWeightsForMatMulNBits' fusion optimization
// is skipped — see graphOptimizationLevel below and STEPS.md Step 1 notes for the full story
// (including why a self-converted int8 alternative was tried and dropped: near-identical output
// quality, extra size, extra complexity).
export const BASE_MODEL: ModelDef = {
  id: 'rinna-japanese-gpt2-xsmall',
  label: 'rinna GPT-2 (xsmall)',
  sourceRepo: 'rinna/japanese-gpt2-xsmall',
  sessionOptions: { graphOptimizationLevel: 'basic' },
}

// "詩" style: rinna/japanese-gpt2-xsmall fine-tuned on public-domain 童謡 lyrics (see
// training-data/douyou/). Self-converted with the same optimum-cli export + quantize_dynamic
// (QOperator int8) pipeline as the base model's self-converted alternative — this one loads
// fine under the default graph optimization level, no session_options workaround needed, since
// QOperator format never triggers the QDQ/MatMulNBits fusion bug the base model's file hits.
export const DOUYOU_MODEL: ModelDef = {
  id: 'rinna-japanese-gpt2-xsmall-douyou',
  label: 'rinna GPT-2 (xsmall) — 童謡ファインチューン',
  sourceRepo: 'rinna/japanese-gpt2-xsmall (fine-tuned)',
}

// "小説" style: rinna/japanese-gpt2-xsmall fine-tuned on public-domain Aozora Bunko works (see
// training-data/aozora/) — same self-converted pipeline as DOUYOU_MODEL, no session_options
// workaround needed for the same reason (QOperator format quantization).
export const AOZORA_MODEL: ModelDef = {
  id: 'rinna-japanese-gpt2-xsmall-aozora',
  label: 'rinna GPT-2 (xsmall) — 青空文庫ファインチューン',
  sourceRepo: 'rinna/japanese-gpt2-xsmall (fine-tuned)',
}

// "名言" style: rinna/japanese-gpt2-xsmall fine-tuned on public-domain historical quotes (see
// training-data/meigen/) — same self-converted pipeline as DOUYOU_MODEL/AOZORA_MODEL.
export const MEIGEN_MODEL: ModelDef = {
  id: 'rinna-japanese-gpt2-xsmall-meigen',
  label: 'rinna GPT-2 (xsmall) — 名言ファインチューン',
  sourceRepo: 'rinna/japanese-gpt2-xsmall (fine-tuned)',
}

// All 3 of Plan.md's fine-tuned styles now exist.
export const MODELS: ModelDef[] = [BASE_MODEL, DOUYOU_MODEL, AOZORA_MODEL, MEIGEN_MODEL]

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
