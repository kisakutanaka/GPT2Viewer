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
- [ ] `git init` して `.gitignore`（node_modules等）を整備し、初回commit（ユーザー担当）
- [ ] GitHubにリモートリポジトリを作成し、push（以降の各Stepはキリの良い単位でこまめにcommit & push、ユーザー担当）

## Step 1: transformers.js 導入 & モデルロード確認

- [ ] `@huggingface/transformers` を依存関係に追加
- [ ] `saldra/rinna-japanese-gpt2-xsmall-onnx` をロードするだけの最小コードを書く（コンソールにモデル情報を出す）
- [ ] ブラウザで実際にモデルダウンロード〜ロードが成功することを確認（キャッシュ・CORSまわりの問題がないか確認）

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

## Step 9: GitHub Pagesへのデプロイ

- [ ] GitHub Actions等でビルド＆Pagesへの自動デプロイを設定
- [ ] 実際にPages上でモデルダウンロード〜生成まで動作することを確認
