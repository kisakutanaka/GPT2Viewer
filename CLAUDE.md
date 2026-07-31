# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

"Transformer文章生成ビジュアライザー" — an educational web app that visualizes how a Transformer-based language model generates text one token at a time (next-token prediction), rather than producing a whole sentence at once. Full requirements are in [Plan.md](Plan.md); the step-by-step build plan and progress notes are in [STEPS.md](STEPS.md) — check it for what's done and what's next before starting work.

- **User flow**: user picks 3 seed words from on-screen buttons → picks 1 of 3 fine-tuned models → watches the model generate text token-by-token in real time, with each step showing the candidate next tokens and their probabilities.
- **Target device**: iPhone (mobile Safari) is the assumed runtime environment, not desktop. Design UI for small touch viewports, and treat mobile Safari's tighter WASM/memory limits as a real constraint when choosing model precision/size, not just a desktop concern.
- **Educational goal**: make next-token prediction and probability-based generation intuitively visible, and let users compare text style across the 3 fine-tuned models.

## Commands

- `npm run dev` — start Vite dev server
- `npm run build` — type-check (`tsc -b`) + production build to `dist/`
- `npm run preview` — serve the production build locally
- `npm run lint` — oxlint

No test suite exists yet.

## Architecture

- **Stack**: Vite + React + TypeScript. Model inference runs client-side via [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) (transformers.js), which wraps ONNX Runtime Web (wasm backend). No backend — this is a static site deployed to GitHub Pages.
- **Deploy**: `.github/workflows/deploy.yml` builds and deploys to GitHub Pages automatically on every push to `main` (set up early, per project preference — checking Pages after each change is part of the normal workflow, not just a final step). One-time manual setup required on GitHub: Settings → Pages → Source → "GitHub Actions". `vite.config.ts` sets `base: '/GPT2Viewer/'` to match the GitHub Pages subpath — update this if the repo is ever renamed.
- **Model loading** (`src/lib/model.ts`): the base model is `rinna/japanese-gpt2-xsmall` (MIT license, GPT2 architecture, 6 layers, 512 hidden, sentencepiece/T5Tokenizer). Its ONNX build, `saldra/rinna-japanese-gpt2-xsmall-onnx` (also MIT), is copied locally under `public/models/rinna-japanese-gpt2-xsmall/` (the quantized ONNX file renamed `onnx/model_quantized.onnx` to match transformers.js's default naming) — it is **not** loaded from the Hugging Face hub at runtime (`env.allowRemoteModels = false`; `env.localModelPath` is derived from Vite's `import.meta.env.BASE_URL` so it resolves correctly under the `/GPT2Viewer/` Pages subpath).
  - The quantized graph (int4 blockwise/`MatMulNBits`) fails to load in onnxruntime-web/wasm by default (`Missing required scale: ...wte.weight_merged_0_scale for node: ...DequantizeLinear`) despite loading fine under `onnxruntime-node` — a real trap since Node-only testing looks like success. Root cause (best understanding, not fully source-verified): GPT2 ties `wte.weight` and the LM head to the same tensor; ORT's initializer-dedup pass renames the merged tensor, but the QDQ→`MatMulNBits` fusion pass (`TransposeDQWeightsForMatMulNBits`, part of the `'extended'`/`'all'` optimization levels) then looks for a scale initializer under the renamed name and doesn't find it. **Fix**: pass `session_options: { graphOptimizationLevel: 'basic' }` to `from_pretrained` — this skips that fusion pass entirely and the exact same 40MB file loads fine. This is required whenever loading this specific model; don't drop it.
  - A self-converted alternative was also tried: conda env `ai-tools` (Python 3.10, generic AI/ML tooling, not project-specific — reuse it rather than creating a new env) with `optimum-onnx` + `onnx` + `onnxruntime` + `torch`. Export: `optimum-cli export onnx --model rinna/japanese-gpt2-xsmall --task text-generation-with-past <dir>` (produces a single merged `model.onnx` matching transformers.js's naming). Quantize: `onnxruntime.quantization.quantize_dynamic(..., weight_type=QuantType.QUInt8)` (standard dynamic int8, avoids the fusion bug entirely since it doesn't produce `MatMulNBits`). Result: 215MB → ~54MB. Compared against saldra's 40MB int4 build (top-10 next-token candidate overlap 6–8/10 across test prompts, no clear quality gap under either greedy or sampled generation) and dropped in favor of the smaller saldra file plus the `graphOptimizationLevel` workaround — less to maintain, no independent conversion pipeline to keep in sync. Revisit this approach if adding the other 2 models turns up a saldra-equivalent pre-converted build that also breaks, or if none exists for a given model.
  - The model's `generation_config` defaults to `do_sample: true` for a reason: greedy decoding (always the argmax token) tends to degenerate into `<unk>`-token loops on this small (43.7M param) model, while top-k + temperature sampling produces clearly more fluent Japanese. Build generation logic (Steps 3–4) around sampling, not greedy, as the default/only mode.
  - 40MB comfortably clears GitHub's "consider Git LFS" push warning threshold (50MB) without hitting it, and is well under the 100MB hard limit. Decision: skip Git LFS even once the other two models are added — but note Git LFS objects aren't served as-is by the GitHub Actions Pages deploy path used here without `lfs: true` on `actions/checkout`, so revisit deliberately if switching later, don't assume it "just works."
- When adding the other two fine-tuned models (Plan.md's 3-model requirement), expect to hit similar per-model ONNX/quantization quirks — check for an existing hub ONNX build first, try the `graphOptimizationLevel: 'basic'` workaround before reaching for a from-scratch conversion, and always verify actual browser loading (not just Node) since Node and wasm ONNX Runtime backends have already diverged once here in what they can load.
- **Generation loop** (`src/lib/generate.ts`): `model.generate()` does not expose per-step logits/scores in this version of transformers.js (traced into the bundled source — `return_dict_in_generate` has a literal `// TODO: scores, // logits` where they'd go), but the app needs per-step candidate tokens + probabilities for the educational UI. So generation is hand-rolled by calling the model's own step-preparation helpers directly — `prepare_inputs_for_generation`, `_update_model_kwargs_for_generation`, `_prepare_generation_config` — the same ones `generate()` uses internally. These are present in the public `.d.ts` (untyped as `any`) but are underscore-prefixed/internal-flavored and not officially documented public API, so a transformers.js upgrade could change their signatures; if generation breaks after a dependency bump, check there first. Selection at each step samples (top-k + temperature) from the softmax distribution rather than taking argmax, per the do_sample/greedy-degeneration note above.

## Design constraints (from Plan.md)

- **Browser-only, no backend** — must be deployable as a static site on GitHub Pages.
- **Simplicity over features** — build the MVP first; do not add functionality beyond what's listed in Plan.md's MVP section. Prioritize educational clarity over UI polish or generality.
- MVP scope: 3-word selection, 3 switchable models, real-time single-token-at-a-time generation, per-step candidate-token/probability display, live-updating generated text.

## Working conventions

- `git add`/`commit`/`push` are handled by the user, not Claude — prepare/stage changes but don't commit or push them.
