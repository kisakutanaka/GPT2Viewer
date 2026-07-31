# 実装ステップ (STEPS.md)

[Plan.md](Plan.md) のMVPを、まず動くものから小さく積み上げるためのステップリスト。
最初のゴールは「rinna/japanese-gpt2-xsmall で1トークンずつ生成する様子が画面に出る」こと。

前提技術メモ:
- TypeScript + Vite（GitHub Pages配信を想定した静的ビルド）
- モデル推論はクライアントサイドで [transformers.js](https://github.com/huggingface/transformers.js)（`@huggingface/transformers`）を使用
- rinna/japanese-gpt2-xsmall（MITライセンス）のONNX変換版（`saldra/rinna-japanese-gpt2-xsmall-onnx`、MITライセンス、40MB、int4 blockwise量子化）を `public/models/` にローカルコピーして配信している。読み込み時に `session_options: { graphOptimizationLevel: 'basic' }` を指定するのが必須（詳細はStep 1参照）
- 動作環境はiPhone（モバイルSafari）を想定。モデルサイズ・メモリ使用量は都度この観点で確認する
- モデルのデフォルト`generation_config`は `do_sample: true` 前提。greedy（常に最尤トークン）で生成すると`<unk>`ループ等に陥りやすく、top-k+温度によるサンプリングの方が明らかに自然な文章になることを確認済み（Step 1の検証時）。Step 3/4はこれを踏まえてサンプリングを前提に設計する

---

## Step 0: プロジェクト雛形 & リポジトリ準備 ✅

- [x] Vite + TypeScript でプロジェクトを初期化（react-tsテンプレート採用）
- [x] GitHub Pages向けに `vite.config.ts` の `base: '/GPT2Viewer/'` を設定（実際のリポジトリ名と異なる場合は要調整）
- [x] `npm run dev` でローカル起動確認
- [x] `git init` して `.gitignore`（node_modules等）を整備し、初回commit（ユーザー担当）
- [x] GitHubにリモートリポジトリを作成し、push（`kisakutanaka/GPT2Viewer`、以降の各Stepはキリの良い単位でこまめにcommit & push、ユーザー担当）

## Step 1: transformers.js 導入 & モデルロード確認 ✅

- [x] `@huggingface/transformers` を依存関係に追加
- [x] モデルをロードするだけの最小コードを書く（`src/lib/model.ts`、コンソールにモデル情報を出す）
- [x] ブラウザで実際にモデルダウンロード〜ロードが成功することを確認（ローカル・本番Pagesとも確認済み、エラーなし）

**わかったこと（経緯）:**
- 当初 `saldra/rinna-japanese-gpt2-xsmall-onnx`（HFの変換済みONNX）をそのまま使おうとしたが、量子化版（`dtype: 'q8'`、int4 blockwise/MatMulNBits形式）が **onnxruntime-web(wasm)ではロード失敗**（`Missing required scale: ...wte.weight_merged_0_scale`。Node/onnxruntime-nodeでは成功していたため気づきにくい差異）
- 原因を調査した結果、GPT2のtied weights（`wte.weight`とlm_headが同一テンソル）に対する重複排除最適化と、QDQ→MatMulNBits融合最適化（`TransposeDQWeightsForMatMulNBits`）の組み合わせがエッジケースを踏んでいる可能性が高いと判明。`session_options: { graphOptimizationLevel: 'basic' }`（`'extended'`/`'all'`で有効な融合パスをスキップ）を指定すれば **元のsaldra版40MBファイルのままロードできる**ことを、`onnxruntime-web`を直接Node.jsから叩いて確認した
- 別解として、conda環境 `ai-tools`（Python 3.10、`optimum-onnx` + `onnx` + `onnxruntime` + `torch`）で `rinna/japanese-gpt2-xsmall` を自前で `optimum-cli export onnx` → `quantize_dynamic`（標準dynamic int8）で変換する方法も試した（215MB→54MB）。saldra版(40MB, int4)と自前版(54MB, int8)で生成品質を比較（top-10候補の一致度6〜8/10、サンプリング生成でも明確な差は見られず）した結果、**品質差はほぼ無いと判断しsaldra版（`graphOptimizationLevel: 'basic'`込み）のみ採用**。自前変換の`ai-tools`環境やコマンドは今後別モデルを変換する際に再利用できる
- モデル一式（`saldra/rinna-japanese-gpt2-xsmall-onnx`からダウンロード、MITライセンス）は `public/models/rinna-japanese-gpt2-xsmall/` にコミット済み（`onnx/model_quantized.onnx`にリネーム）
- 40MBのONNXファイルはGit LFS化せず素のgitで管理する方針（合意済み）。3モデル分（見込み100〜150MB程度）でもこのまま進める

## Step 2: トークナイザー動作確認 ✅

- [x] 適当な日本語テキストを encode → token id 配列 → decode で元に戻るか確認（`src/App.tsx`、サンプル文「今日はいい天気ですね」でラウンドトリップ確認済み）
- [x] token id と対応する文字列（サブワード）をコンソール表示できるようにする

## Step 3: 1ステップ分の推論（次トークン候補と確率） ✅

- [x] 入力テキストに対し、モデルから次トークンのlogitsを取得（`src/lib/generate.ts`、`prepare_inputs_for_generation` + `model.forward`）
- [x] softmaxで確率化し、上位N件の候補トークンと確率をコンソール表示
- [x] 次トークンを選んで入力に追加するロジックを書く（「最も確率の高いトークン」固定ではなく、top-k+温度のサンプリングを採用。理由はStep 1のメモ参照 — greedyだと`<unk>`ループに陥りやすいため）

**実装メモ:** `model.generate()`は各ステップのlogits/scoresを返さない（transformers.js側でTODOのまま未実装）ため使わず、`prepare_inputs_for_generation` / `_update_model_kwargs_for_generation` / `_prepare_generation_config`という内部寄りの公開メソッドを直接呼んで自前の生成ループを組んでいる。ライブラリのバージョンアップで変わる可能性がある点に注意。

## Step 4: 複数ステップの自動生成ループ ✅

- [x] Step 3 のロジックをループさせ、指定トークン数まで自動生成（`src/App.tsx`、20トークンで動作確認）
- [x] 各ステップの「選ばれたトークン」「候補トップN」「確率」を配列として保持（React state `steps`）
- [x] コンソールログで生成過程が段階的に見えることを確認

## Step 5: 最小UI（プレーンなテキスト入力版） ✅

- [x] テキスト入力欄 + 生成開始ボタンのみのシンプルなUIを作成
- [x] 生成ボタン押下で Step 4 のロジックを実行し、生成中の文章をリアルタイムに画面表示
- [x] 各ステップの候補トークン・確率も画面上に表示（リスト形式、装飾は後回し）

## Step 6: 単語選択UIへの差し替え ✅

- [x] Plan.mdの仕様通り、自由入力ではなく「画面に並ぶ単語ボタンから3つ選ぶ」UIに変更（`WORD_CHOICES`、12語から3つトグル選択）
- [x] 選んだ3単語から初期プロンプトを組み立てて生成開始（選択順に連結）

## Step 7: モデル切り替えの土台づくり（複数モデル対応の準備） ✅（1モデル分のみ、残り2つは未着手）

- [x] モデル読み込み処理を、モデルID/名前を差し替え可能な形に整理（`ModelDef` + `MODELS`配列、`loadModel(def)`）
- [x] とりあえず現状の1モデルのみを選択肢として表示するUI（プルダウン）を用意。切り替え時は選択モデルを再ロード
- [ ] 他2モデルは後続タスクで追加（別モデルのONNX変換 or 対応モデル探し。Step 1の手順・`ai-tools`環境が使える）

## Step 8: UI/UX仕上げ ✅（クレジット表記のみ未対応）

Plan.mdのUXを3シーン構成で実装:
- **シーン1**（`WordSelectStage`）: タイトル「さくぶんAI」→ 単語を3つ選択 → 文体（小説/名言/詩、現状は3つとも同じ1モデル）を選択 → 「作文する」
- **シーン2**（`GenerationStage`）: 1トークンずつ「候補と確率バー表示 → 選ばれたトークンをハイライト → 文章に追加」を繰り返す。2段階の間（`CANDIDATE_DISPLAY_MS` / `CHOSEN_HOLD_MS`、`src/App.tsx`冒頭）でユーザーが確率を見てから選択を確認できるようにペース調整（ユーザー側で750ms/750ms、50トークンに調整済み）
- **シーン3**（`ResultStage`）: 完成した文章を表示 →「もう一度体験する」でシーン1に戻る（モデルはキャッシュ済みで再ロードなし）

- [x] 生成中トークンのハイライトや確率バーなど、教育効果を高める見せ方を追加
- [x] 不要な機能を足していないか、Plan.mdのMVPスコープと突き合わせて確認（旧Viteテンプレートの未使用assets/icons.svgも削除）
- [ ] 使用モデル（rinna/japanese-gpt2-xsmall、MITライセンス）のクレジット表記をフッターかREADMEに追加

**わかったこと（Step 8中に見つけた生成ロジックの改善）:**
- `<unk>`等の非EOS特殊トークン（id 0,1,3,4,5,6）は候補表示はするが、実際の出力候補（サンプリング対象）からは除外（`src/lib/generate.ts`の`NON_EOS_SPECIAL_TOKEN_IDS`）
- 空文字にデコードされるトークン（sentencepieceの語頭マーカー単体など）も同様に実際の出力候補から除外（`isSelectablePiece`）。理由: 選ばれても画面上の生成文が視覚的に変化せず、シーン2の「文章が育つ」演出が止まって見えるため

## Step 9: GitHub Pagesへのデプロイ ✅（前倒しで設定済み）

- [x] GitHub Actionsでビルド＆Pagesへの自動デプロイを設定（`.github/workflows/deploy.yml`、mainへのpushで自動実行）
- [x] リポジトリのSettings → Pages → Source を「GitHub Actions」に変更（ユーザー担当、初回のみ）
- [x] 実際にPages上でモデルダウンロード〜ロードまで動作することを確認（自作量子化モデル、エラーなし）

進行状況を都度Pages上で確認したいとのことなので、Step 9は前倒しで設定済み。以降の各Stepはpushするたびに `https://kisakutanaka.github.io/GPT2Viewer/` に自動反映される。
