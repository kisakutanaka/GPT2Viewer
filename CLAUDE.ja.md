# CLAUDE.ja.md

これは [CLAUDE.md](CLAUDE.md) の日本語訳です。Claude Code が参照する正本は英語版の CLAUDE.md であり、このファイルは人間が読むための参考訳です。内容を更新する場合は両方のファイルを合わせてください。

## このプロジェクトについて

「Transformer文章生成ビジュアライザー」— Transformerベースの言語モデルが文章全体を一度に生成しているのではなく、「次に来るトークン」を1つずつ予測しながら生成している仕組みを可視化する教育用Webアプリ。詳細な要件は [Plan.md](Plan.md)、実装のステップリストと進捗メモは [STEPS.md](STEPS.md) を参照。作業前に何が完了していて次に何をするかSTEPS.mdで確認すること。

- **ユーザーフロー**: 3シーン構成（`WordSelectStage` → `GenerationStage` → `ResultStage`、`src/App.tsx`がオーケストレーション）。単語を3つ選択 → 文体（小説/名言/詩。現状は3つとも同じ1モデルにマッピング、STEPS.mdのStep 7参照）を選択 → AIが文章を1トークンずつリアルタイムに生成する様子を観察（各ステップで候補トークンと確率を表示してから、選ばれたトークンがハイライトされて文章に追加される）→ 結果画面（もう一度体験するボタン付き）。
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
- **モデルロード**（`src/lib/model.ts`）: ベースモデルは `rinna/japanese-gpt2-xsmall`（MITライセンス、GPT2アーキテクチャ、6層・512隠れ層、sentencepiece/T5Tokenizer）。そのONNX変換版 `saldra/rinna-japanese-gpt2-xsmall-onnx`（同じくMITライセンス）をローカルにコピーし、`public/models/rinna-japanese-gpt2-xsmall/` にコミット済み（量子化ONNXファイルはtransformers.js標準命名に合わせて `onnx/model_quantized.onnx` にリネーム）。**実行時にHugging Face Hubからは読み込まない**（`env.allowRemoteModels = false`。`env.localModelPath` はViteの `import.meta.env.BASE_URL` から組み立てているので `/GPT2Viewer/` のPagesサブパス配下でも正しく解決する）。
  - この量子化版（int4 blockwise/`MatMulNBits`形式）はデフォルトだと **onnxruntime-web(wasm)ではロードに失敗する**（`Missing required scale: ...wte.weight_merged_0_scale for node: ...DequantizeLinear`）。onnxruntime-node（Node.js）では問題なくロードできてしまうため、Nodeだけのテストでは気づきにくい落とし穴だった。原因は（ソースまで完全に裏取りはしていないが）おそらく: GPT2は`wte.weight`とLM headが同一テンソル（tied weights）であり、ONNX Runtimeの重複排除最適化がこのテンソルをリネームする一方、QDQ→`MatMulNBits`融合パス（`TransposeDQWeightsForMatMulNBits`、`'extended'`/`'all'`最適化レベルで有効）がリネーム後の名前でスケール初期化子を探しに行って見つからない、という食い違い。**対処法**: `from_pretrained` に `session_options: { graphOptimizationLevel: 'basic' }` を渡すことで、この融合パス自体をスキップすれば同じ40MBファイルがそのままロードできる。このモデルを読み込む際は必ず必要な設定なので外さないこと。
  - 別案として自前変換も試した: conda環境 `ai-tools`（Python 3.10、プロジェクト専用ではなくAI/ML作業全般向け — 新規作成せず再利用すること）に `optimum-onnx` + `onnx` + `onnxruntime` + `torch`。変換: `optimum-cli export onnx --model rinna/japanese-gpt2-xsmall --task text-generation-with-past <dir>`（transformers.js標準命名と一致する単一の `model.onnx` が出力される）。量子化: `onnxruntime.quantization.quantize_dynamic(..., weight_type=QuantType.QUInt8)`（標準dynamic int8、`MatMulNBits`を生成しないため上記バグを踏まない）。結果: 215MB→約54MB。saldra版(40MB, int4)と比較（プロンプトごとの次トークン候補トップ10の一致度6〜8/10、greedy/サンプリングいずれでも明確な品質差なし）した結果、より小さいsaldra版＋`graphOptimizationLevel`回避策を採用し、自前変換は見送った（独自の変換パイプラインを維持するコストに見合わないため）。残り2モデルでsaldra版相当の変換済みONNXが存在しない、または同様に壊れている場合はこの自前変換の手順を再利用する。
  - このモデルの`generation_config`は`do_sample: true`が既定値になっている。greedy（常に最尤トークンを選ぶ）で生成すると`<unk>`ループなどに陥りやすく、top-k＋温度によるサンプリングの方が明らかに自然な日本語になることを確認済み。Step 3〜4の生成ロジックはgreedyではなくサンプリングを前提に設計すること。
  - 40MBはGitHubの「50MB超はGit LFS推奨」警告の閾値には達しておらず、100MBのハード上限にも余裕がある。残り2モデルを追加してもLFS化は見送る方針（合意済み）。なお、この構成（GitHub Actions経由のPagesデプロイ）はGit LFSのオブジェクトをそのままでは配信できず、`actions/checkout` に `lfs: true` を付ける等の追加対応が要る点に注意（「LFSにすれば自然に動く」ではない）。
- 残り2モデル（Plan.mdの3モデル要件）を追加する際は、同様のONNX/量子化まわりの癖に当たる可能性が高い。まずHub上に既存のONNX変換版がないか探し、`graphOptimizationLevel: 'basic'`で回避できないか試してから自前変換に進むこと。またNode.jsではなく実際のブラウザでのロードをモデルごとに都度確認すること（Node/wasmのONNX Runtimeバックエンドの対応状況が既に一度乖離しているため）。
- **生成ループ**（`src/lib/generate.ts`）: このバージョンのtransformers.jsでは `model.generate()` が各ステップのlogits/scoresを返さない（バンドルされたソースを実際に追跡すると、`return_dict_in_generate`のところに文字通り `// TODO: scores, // logits` とコメントされたまま未実装になっている）。一方このアプリの教育的UIには各ステップの候補トークンと確率が必要なため、生成ループは自前で組んでいる — `generate()`が内部で使っているのと同じ `prepare_inputs_for_generation` / `_update_model_kwargs_for_generation` / `_prepare_generation_config` を直接呼び出す方式。これらは公開の `.d.ts` には存在する（型は`any`）が、アンダースコア始まりなど内部寄りの性質で正式に文書化された公開APIではないため、transformers.jsのアップデートでシグネチャが変わる可能性がある。依存関係更新後に生成が壊れたら、まずここを疑うこと。各ステップの選択は上記の do_sample/greedy崩壊の理由により、argmaxではなくtop-k＋温度によるサンプリングを採用している。
  - `stepGeneration`は同じsoftmax分布から2種類のリストを作る: `candidates`（生の上位K件、フィルタなしでそのままUIに表示 — 透明性のため）と `selectable`（非EOS特殊トークン=id 0,1,3,4,5,6、つまり`</s>`以外の全特殊トークンと、空文字にデコードされるトークン=孤立したsentencepiece語頭マーカー等、を除外した上位K件）。実際にサンプリングして次トークンを選ぶのは`selectable`からのみ。理由: 生成文中に生の`<unk>`/`[PAD]`/空文字トークンが出てもただのノイズで学びがなく、空文字トークンが選ばれるとシーン2の画面上の文章が視覚的に変化せず不具合に見えてしまうため。候補フィルタリングロジックを他で追加する場合も、「表示は全部見せる、サンプリング対象だけ絞る」というこの分離を踏襲すること。
- **UI構成**: `src/App.tsx`が全状態（`scene`、選択単語/文体、生成の進行状況）とモデルキャッシュ（`modelCacheRef`、`Map<modelId, {model, tokenizer}>` — 文体切り替えや再体験時にロード済みモデルを再ロードしないため）を持つ。各シーンは`src/components/`配下の表示用コンポーネント: `WordSelectStage.tsx`, `GenerationStage.tsx`, `ResultStage.tsx`。シーン2の1トークンごとの見せ方はあえて2段階（`App.tsx`冒頭の`CANDIDATE_DISPLAY_MS` / `CHOSEN_HOLD_MS`定数）— まず候補と確率バーを何もハイライトせずに表示し、その後サンプリングされたトークンをハイライトして少し止め、それから文章に追加する、という流れにすることで「確率→選択」の瞬間が一瞬で流れずに見えるようにしている。生成のペース・長さを変更する依頼が来たら、まずこの2定数と`MAX_NEW_TOKENS`を見ること。

## 設計方針（Plan.mdより）

- **ブラウザのみで動作、バックエンドなし** — GitHub Pages上で静的サイトとしてデプロイ可能であること。
- **機能より単純さを優先** — まずはMVPを完成させる。Plan.mdのMVPセクションに記載のない機能は追加しない。UIの作り込みや汎用性より、教育的な分かりやすさを優先する。
- MVPスコープ: 3単語選択、3モデル切り替え、1トークンずつのリアルタイム生成、各ステップでの候補トークン・確率表示、生成文章のリアルタイム表示。

## 作業上の約束事

- `git add`/`commit`/`push` はユーザーが行う。Claudeは変更の準備までにとどめ、commit/pushは行わない。
