# 実装ステップ (STEPS.md)

[Plan.md](Plan.md) のMVPを、まず動くものから小さく積み上げるためのステップリスト。
最初のゴールは「rinna/japanese-gpt2-xsmall で1トークンずつ生成する様子が画面に出る」こと。

前提技術メモ:
- TypeScript + Vite（GitHub Pages配信を想定した静的ビルド）
- モデル推論はクライアントサイドで [transformers.js](https://github.com/huggingface/transformers.js)（`@huggingface/transformers`）を使用
- rinna/japanese-gpt2-xsmall はそのままだとONNXがないため、変換済みの `saldra/rinna-japanese-gpt2-xsmall-onnx`（onnx/フォルダ・tokenizer.json・spiece.model 完備）を利用する

---

## Step 0: プロジェクト雛形 & リポジトリ準備 ✅

- [x] Vite + TypeScript でプロジェクトを初期化（react-tsテンプレート採用）
- [x] GitHub Pages向けに `vite.config.ts` の `base: '/GPT2Viewer/'` を設定（実際のリポジトリ名と異なる場合は要調整）
- [x] `npm run dev` でローカル起動確認
- [x] `git init` して `.gitignore`（node_modules等）を整備し、初回commit（ユーザー担当）
- [x] GitHubにリモートリポジトリを作成し、push（`kisakutanaka/GPT2Viewer`、以降の各Stepはキリの良い単位でこまめにcommit & push、ユーザー担当）

## Step 1: transformers.js 導入 & モデルロード確認 ✅

- [x] `@huggingface/transformers` を依存関係に追加
- [x] `saldra/rinna-japanese-gpt2-xsmall-onnx` をロードするだけの最小コードを書く（`src/lib/model.ts`、コンソールにモデル情報を出す）
- [x] ブラウザで実際にモデルダウンロード〜ロードが成功することを確認（CORSエラーなし）

**わかったこと / 積み残し:**
- このリポジトリは transformers.js 標準の `onnx/model.onnx` 命名ではなく optimum-cli 由来の `decoder_model_merged*.onnx` 命名なので `model_file_name: 'decoder_model_merged'` の指定が必要
- `dtype: 'q8'`（量子化版 `decoder_model_merged_quantized.onnx`, 40MB）は QDQ/MatMulNBits 形式が原因で **onnxruntime-web(wasm)ではロード失敗**（Node/onnxruntime-nodeでは成功していたため気づきにくい差異）。当面 `dtype: 'fp32'`（非量子化, 152MB）で進める
- 後日タスク: モデルサイズが大きい（152MB×モデル数）ので、量子化ロード問題の解決（別の量子化形式で再変換 or onnxruntime-webのアップデート待ち）をどこかのStepで検討する

## Step 2: トークナイザー動作確認

- [ ] 適当な日本語テキストを encode → token id 配列 → decode で元に戻るか確認
- [ ] token id と対応する文字列（サブワード）をコンソール表示できるようにする

## Step 3: 1ステップ分の推論（次トークン候補と確率）

- [ ] 入力テキストに対し、モデルから次トークンのlogitsを取得
- [ ] softmaxで確率化し、上位N件の候補トークンと確率をコンソール表示
- [ ] 最も確率の高いトークンを1つ選んで入力に追加するロジックを書く

## Step 4: 複数ステップの自動生成ループ

- [ ] Step 3 のロジックをループさせ、指定トークン数まで自動生成
- [ ] 各ステップの「選ばれたトークン」「候補トップN」「確率」を配列として保持
- [ ] コンソールログで生成過程が段階的に見えることを確認

## Step 5: 最小UI（プレーンなテキスト入力版）

- [ ] テキスト入力欄 + 生成開始ボタンのみのシンプルなUIを作成
- [ ] 生成ボタン押下で Step 4 のロジックを実行し、生成中の文章をリアルタイムに画面表示
- [ ] 各ステップの候補トークン・確率も画面上に表示（表 or リスト形式でOK、装飾は後回し）

## Step 6: 単語選択UIへの差し替え

- [ ] Plan.mdの仕様通り、自由入力ではなく「画面に並ぶ単語ボタンから3つ選ぶ」UIに変更
- [ ] 選んだ3単語から初期プロンプトを組み立てて生成開始

## Step 7: モデル切り替えの土台づくり（複数モデル対応の準備）

- [ ] モデル読み込み処理を、モデルID/名前を差し替え可能な形に整理
- [ ] とりあえず現状の1モデルのみを選択肢として表示するUI（プルダウン等）を用意
- [ ] 他2モデルは後続タスクで追加（別モデルのONNX変換 or 対応モデル探し）

## Step 8: UI/UX仕上げ

- [ ] 生成中トークンのハイライトや確率バーなど、教育効果を高める見せ方を追加
- [ ] 不要な機能を足していないか、Plan.mdのMVPスコープと突き合わせて確認
- [ ] 使用モデル（rinna/japanese-gpt2-xsmall、MITライセンス）のクレジット表記をフッターかREADMEに追加

## Step 9: GitHub Pagesへのデプロイ ✅（前倒しで設定済み）

- [x] GitHub Actionsでビルド＆Pagesへの自動デプロイを設定（`.github/workflows/deploy.yml`、mainへのpushで自動実行）
- [ ] リポジトリのSettings → Pages → Source を「GitHub Actions」に変更（ユーザー担当、初回のみ）
- [ ] 実際にPages上でモデルダウンロード〜生成まで動作することを確認

進行状況を都度Pages上で確認したいとのことなので、Step 9は前倒しで設定済み。以降の各Stepはpushするたびに `https://kisakutanaka.github.io/GPT2Viewer/` に自動反映される。
- [ ] 実際にPages上でモデルダウンロード〜生成まで動作することを確認
