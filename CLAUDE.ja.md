# CLAUDE.ja.md

これは [CLAUDE.md](CLAUDE.md) の日本語訳です。Claude Code が参照する正本は英語版の CLAUDE.md であり、このファイルは人間が読むための参考訳です。内容を更新する場合は両方のファイルを合わせてください。

## このプロジェクトについて

「Transformer文章生成ビジュアライザー」— Transformerベースの言語モデルが文章全体を一度に生成しているのではなく、「次に来るトークン」を1つずつ予測しながら生成している仕組みを可視化する教育用Webアプリ。詳細な要件は [Plan.md](Plan.md)、実装のステップリストと進捗メモは [STEPS.md](STEPS.md) を参照。作業前に何が完了していて次に何をするかSTEPS.mdで確認すること。

- **ユーザーフロー**: 画面上の単語ボタンから3つの単語を選択 → 3種類のファインチューニング済みモデルから1つを選択 → AIが文章を1トークンずつリアルタイムに生成する様子を観察する。各ステップで次トークンの候補とその確率が表示される。
- **想定端末**: デスクトップではなくiPhone（モバイルSafari）を動作環境として想定する。UIは小さいタッチビューポート向けに設計し、モバイルSafariのWASM/メモリ制限が厳しいことをデスクトップだけでなく実際の制約として扱うこと（モデルの精度・サイズ選定に直結する）。
- **教育目標**: 次トークン予測と確率に基づく文章生成の過程を直感的に見えるようにし、3種類のモデルによる文章スタイルの違いを比較できるようにする。

## コマンド

- `npm run dev` — Vite開発サーバー起動
- `npm run build` — 型チェック（`tsc -b`）＋本番ビルド（`dist/`へ出力）
- `npm run preview` — 本番ビルドをローカルで確認
- `npm run lint` — oxlint

テストスイートはまだ無し。

## アーキテクチャ

- **スタック**: Vite + React + TypeScript。モデル推論はクライアントサイドで [`@huggingface/transformers`](https://github.com/huggingface/transformers.js)（transformers.js、内部はONNX Runtime Webのwasmバックエンド）を使用。バックエンドなしのGitHub Pages静的サイト。
- **デプロイ**: `.github/workflows/deploy.yml` が `main` へのpushのたびに自動ビルド＆Pagesデプロイを行う（最終ステップではなく早い段階で設定済み — 都度Pages上で確認するのが通常のワークフロー）。GitHub側で一度だけ手動設定が必要: Settings → Pages → Source を「GitHub Actions」に変更。`vite.config.ts` の `base: '/GPT2Viewer/'` はGitHub Pagesのサブパスに合わせたもので、リポジトリ名が変わった場合は要更新。
- **モデルロード**（`src/lib/model.ts`）: `rinna/japanese-gpt2-xsmall`（MITライセンス、GPT2アーキテクチャ、6層・512隠れ層、sentencepiece/T5Tokenizer）をONNX変換した `saldra/rinna-japanese-gpt2-xsmall-onnx` を読み込む。このリポジトリ固有で分かりにくい点が2つあり、別モデルに差し替える際も踏みやすい落とし穴:
  - transformers.js標準の `onnx/model.onnx` 命名ではなく、`optimum-cli export onnx` 由来の生の命名（`decoder_model_merged*.onnx`）のままなので、`from_pretrained` に `model_file_name: 'decoder_model_merged'` を明示的に渡す必要がある。
  - 量子化版（`dtype: 'q8'`、`decoder_model_merged_quantized.onnx`、約40MB）はQDQ/MatMulNBits量子化形式のため、**onnxruntime-web(wasm)ではロードに失敗する**（`Missing required scale for ... wte.weight`）。onnxruntime-node（Node.js）では問題なくロードできていたため気づきにくい差異。現状は回避策として `dtype: 'fp32'`（非量子化、約152MB）を使用しており未解決。iPhone向けを考えるとファイルサイズは無視できない懸念点。
- 残り2モデル（Plan.mdの3モデル要件）を追加する際も、モデルごとに同様のONNX変換・ファイル命名・量子化まわりの癖に当たる可能性が高い。Node.jsではなく実際のブラウザでのロードを都度確認すること（Node/wasmのONNX Runtimeバックエンドの対応状況が既に一度乖離しているため）。

## 設計方針（Plan.mdより）

- **ブラウザのみで動作、バックエンドなし** — GitHub Pages上で静的サイトとしてデプロイ可能であること。
- **機能より単純さを優先** — まずはMVPを完成させる。Plan.mdのMVPセクションに記載のない機能は追加しない。UIの作り込みや汎用性より、教育的な分かりやすさを優先する。
- MVPスコープ: 3単語選択、3モデル切り替え、1トークンずつのリアルタイム生成、各ステップでの候補トークン・確率表示、生成文章のリアルタイム表示。

## 作業上の約束事

- `git add`/`commit`/`push` はユーザーが行う。Claudeは変更の準備までにとどめ、commit/pushは行わない。
