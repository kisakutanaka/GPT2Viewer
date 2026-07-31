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
- **Model loading** (`src/lib/model.ts`): the base model is `rinna/japanese-gpt2-xsmall` (MIT license, GPT2 architecture, 6 layers, 512 hidden, sentencepiece/T5Tokenizer), self-converted to ONNX and committed under `public/models/rinna-japanese-gpt2-xsmall/` — it is **not** loaded from the Hugging Face hub at runtime (`env.allowRemoteModels = false`; `env.localModelPath` is derived from Vite's `import.meta.env.BASE_URL` so it resolves correctly under the `/GPT2Viewer/` Pages subpath).
  - Why self-converted: the hub's only pre-existing ONNX build (`saldra/rinna-japanese-gpt2-xsmall-onnx`) has a quantized variant using a QDQ/MatMulNBits scheme that fails to load in onnxruntime-web/wasm (`Missing required scale for ... wte.weight`) despite loading fine under `onnxruntime-node` — a real trap since Node-only testing looks like success. Its non-quantized fallback is 152MB, a real concern given the iPhone target.
  - How it was produced: a conda env named `ai-tools` (Python 3.10, generically for AI/ML tooling, not project-specific) has `optimum-onnx` + `onnx` + `onnxruntime` + `torch` installed. Export: `optimum-cli export onnx --model rinna/japanese-gpt2-xsmall --task text-generation-with-past <dir>` (produces a single merged `model.onnx`, matching transformers.js's default naming — no `model_file_name` override needed). Quantize: `onnxruntime.quantization.quantize_dynamic(..., weight_type=QuantType.QUInt8)` (standard dynamic int8, not the blockwise/QDQ scheme that broke in wasm). Result: 215MB → ~54MB, loads cleanly in onnxruntime-web with `dtype: 'q8'`. The unquantized intermediate `model.onnx` is not committed (unused at runtime, just bloat).
  - The committed 54MB ONNX file trips GitHub's "consider Git LFS" warning on push (recommended threshold is 50MB) but is well under the 100MB hard limit, so pushes succeed. Decision: skip Git LFS for now even once the other two models are added (~150–300MB total repo size estimated) — note that Git LFS objects aren't served as-is by the GitHub Actions Pages deploy path used here without `lfs: true` on `actions/checkout`, so revisit this deliberately if switching later, don't assume it "just works."
- When adding the other two fine-tuned models (Plan.md's 3-model requirement), expect to repeat this same export+quantize process, and re-verify actual browser loading (not just Node) for each — Node and wasm ONNX Runtime backends have already diverged once here in what they can load.

## Design constraints (from Plan.md)

- **Browser-only, no backend** — must be deployable as a static site on GitHub Pages.
- **Simplicity over features** — build the MVP first; do not add functionality beyond what's listed in Plan.md's MVP section. Prioritize educational clarity over UI polish or generality.
- MVP scope: 3-word selection, 3 switchable models, real-time single-token-at-a-time generation, per-step candidate-token/probability display, live-updating generated text.

## Working conventions

- `git add`/`commit`/`push` are handled by the user, not Claude — prepare/stage changes but don't commit or push them.
