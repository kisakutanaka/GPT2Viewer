import { AutoModelForCausalLM, AutoTokenizer } from '@huggingface/transformers'

// rinna/japanese-gpt2-xsmall (MIT license) converted to ONNX for browser use.
// https://huggingface.co/saldra/rinna-japanese-gpt2-xsmall-onnx
export const MODEL_ID = 'saldra/rinna-japanese-gpt2-xsmall-onnx'

export async function loadModel() {
  const tokenizer = await AutoTokenizer.from_pretrained(MODEL_ID)
  const model = await AutoModelForCausalLM.from_pretrained(MODEL_ID, {
    // q8 (decoder_model_merged_quantized.onnx) uses a QDQ/MatMulNBits quantization
    // that onnxruntime-web's wasm backend fails to load (missing scale for wte weight).
    // Fall back to fp32 (decoder_model_merged.onnx, non-quantized) for now.
    dtype: 'fp32',
    // This repo keeps the raw optimum-cli export names (decoder_model_merged*.onnx)
    // instead of transformers.js's usual onnx/model.onnx convention.
    model_file_name: 'decoder_model_merged',
  })
  return { tokenizer, model }
}
