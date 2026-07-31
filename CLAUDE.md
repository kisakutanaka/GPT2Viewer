# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project status

This repository currently contains only [Plan.md](Plan.md) — no source code, build tooling, or dependencies exist yet. There are no build/lint/test commands to run because nothing has been scaffolded. When starting implementation, set up the project structure per the design direction below and update this file with the actual commands (dev server, build, lint, test) once they exist.

## What this project is

"Transformer文章生成ビジュアライザー" — an educational web app that visualizes how a Transformer-based language model generates text one token at a time (next-token prediction), rather than producing a whole sentence at once. Full requirements are in [Plan.md](Plan.md); key points:

- **User flow**: user picks 3 seed words from on-screen buttons → picks 1 of 3 fine-tuned models → watches the model generate text token-by-token in real time, with each step showing the candidate next tokens and their probabilities.
- **Educational goal**: make next-token prediction and probability-based generation intuitively visible, and let users compare text style across the 3 fine-tuned models.

## Design constraints (from Plan.md)

- **Browser-only, no backend** — must be deployable as a static site on GitHub Pages. Model inference (likely via a JS/WASM runtime such as transformers.js/ONNX Runtime Web, or similar) happens client-side.
- **TypeScript** is the base language.
- **Simplicity over features** — build the MVP first; do not add functionality beyond what's listed in Plan.md's MVP section. Prioritize educational clarity over UI polish or generality.
- MVP scope: 3-word selection, 3 switchable models, real-time single-token-at-a-time generation, per-step candidate-token/probability display, live-updating generated text.
