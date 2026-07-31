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
- **Model loading** (`src/lib/model.ts`): loads `saldra/rinna-japanese-gpt2-xsmall-onnx`, an ONNX conversion of `rinna/japanese-gpt2-xsmall` (MIT license, GPT2 architecture, 6 layers, 512 hidden, sentencepiece/T5Tokenizer). Two things about this specific repo are non-obvious and easy to get wrong if switching models:
  - It keeps the raw `optimum-cli export onnx` file names (`decoder_model_merged*.onnx`) rather than transformers.js's usual `onnx/model.onnx` convention, so `model_file_name: 'decoder_model_merged'` must be passed explicitly to `from_pretrained`.
  - The quantized variant (`dtype: 'q8'`, `decoder_model_merged_quantized.onnx`, ~40MB) uses a QDQ/MatMulNBits quantization scheme that **fails to load in onnxruntime-web/wasm** (`Missing required scale for ... wte.weight`), even though it loads fine under `onnxruntime-node`. Currently using `dtype: 'fp32'` (~152MB, non-quantized) as a workaround. This is unresolved and worth revisiting — the file size is a real concern given the iPhone target.
- When adding the other two fine-tuned models (Plan.md's 3-model requirement), expect to hit similar ONNX-conversion/file-naming/quantization quirks per model — verify actual browser loading (not just Node) for each one, since Node and wasm ONNX Runtime backends have diverged in what they can load at least once already here.

## Design constraints (from Plan.md)

- **Browser-only, no backend** — must be deployable as a static site on GitHub Pages.
- **Simplicity over features** — build the MVP first; do not add functionality beyond what's listed in Plan.md's MVP section. Prioritize educational clarity over UI polish or generality.
- MVP scope: 3-word selection, 3 switchable models, real-time single-token-at-a-time generation, per-step candidate-token/probability display, live-updating generated text.

## Working conventions

- `git add`/`commit`/`push` are handled by the user, not Claude — prepare/stage changes but don't commit or push them.
