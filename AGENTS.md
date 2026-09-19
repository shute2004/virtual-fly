# AGENTS.md

このリポジトリで作業するAIエージェント・開発者向けのプロジェクト規約。

## 1. プロジェクトの核心

`virtual-fly` は「ハエのコネクトームを使った人工ニューラルネットワーク」ではない。

成体オスのショウジョウバエCNSコネクトームを初期状態として、PC上で神経活動・神経修飾・可塑性を時間発展させ、身体・感覚・環境と閉ループ接続する。

コネクトームは固定グラフではなく `t = 0` のスナップショットである。

## 2. やってはいけないこと

明示的な設計変更なしに以下を導入してはいけない。

- 誤差逆伝播
- 勾配降下
- Transformer
- Q-learning
- policy gradient
- actor-critic
- 外部NNによる行動選択
- Flappy Birdの攻略ロジック
- `reward` 数値を直接使ったシナプス更新
- 行動ラベルを予測する分類器

配列・疎行列・GPUカーネルを計算最適化のために使うこと自体は禁止しない。禁止しているのは、それらを通常のNN学習機構として設計することである。

## 3. 学習の意味

学習は次の経路でのみ起こす。

```text
sensory stimulation
      ↓
neural dynamics
      ↓
behavior
      ↓
neuromodulatory stimulation
      ↓
local plasticity
      ↓
changed nervous system
```

外部コードが「正解の重み」を計算して書き込んではならない。

## 4. 実測と仮定を分ける

コード・設定・ドキュメントで、可能な限り次を区別する。

- `observed`
- `literature`
- `inferred`
- `assumed`
- `calibrated`

未知の値を仮定すること自体は許容する。仮定を実測値のように扱うことは禁止する。

### 4.1 現象を「同じ答えを返すアルゴリズム」で置換しない

既知の生物学的・物理的な局所過程がある場合、その過程の結果と似た出力を返す便利な特徴抽出・集約・分類アルゴリズムで代替してはいけない。

目標は、結果だけを再現することではなく、その結果を生じさせる局所過程を時間発展させることである。

感覚系では特に次を守る。

良い例:

```text
local physical stimulus
      ↓
local sensory transduction
      ↓
current into corresponding sensory neuron
      ↓
measured CNS connectivity
      ↓
downstream neural processing
```

悪い例:

```text
camera image
      ↓
external edge / motion / object detector
      ↓
spatial averaging or feature pooling
      ↓
precomputed answer injected into downstream neurons
```

平均・pooling・特徴量化そのものを一律禁止するわけではない。実際の生体回路で対応する統合過程がある場合は、その回路・局所モデルとして実装する。CNSへ入る前に外部コードで意味情報を圧縮してはならない。

対応関係が不明な場合は、推測で全体平均などへ逃げず、公式データ・文献・実接続から解決する。解決できない部分は `assumed` / `calibrated` として局所的な境界に隔離し、空間情報など既知の情報を破壊しない。

## 5. 外部モデル

FlyBody / FlyGym / NeuroMechFly / MuJoCo等を使用してよい。

ただし、付属するRL policyやNN controllerをvirtual-flyのCNSの代わりに使ってはいけない。

利用目的は主として:

- 身体形状
- 物理
- 感覚シミュレーション
- アクチュエータ
- 可視化
- 検証用データ

である。

## 6. 神経→身体変換

暫定adapterを作る場合も、adapterに意思決定を入れない。

良い例:

```text
motor neuron activity -> corresponding actuator drive
```

悪い例:

```text
if obstacle_is_above: flap_down()
```

## 7. コネクトームを破壊しない

MaleCNSのsource snapshotは読み取り専用。

可塑性による変更は仮想個体のmutable stateまたはdeltaとして保存する。

## 8. 最適化より検証を優先する

高速化の前に、小規模回路で参照実装との一致を確認する。

高速backendが結果を変える場合、速度より原因究明を優先する。

## 9. 失敗結果を隠さない

このプロジェクトの目的は「ハエに必ずFlappy Birdを解かせる」ことではない。

学習が起きない、活動が停止する、暴走する、行動が改善しない、といった結果も重要な実験結果として扱う。

結果を良くするために隠れた制御器を追加してはいけない。

## 10. ドキュメント

公開向けドキュメントは英語をdefaultとする。READMEなど主要な公開入口には、日本語・簡体字中国語版も用意することが望ましい。

内部の研究・開発記録は、元の文脈を保持するため日本語で書いてよい。historical recordを見た目の統一だけを目的に一括翻訳しない。

API名、固有名詞、コード識別子は必要に応じて英語を使用する。

設計を変えた場合は、コードだけでなく関連する `docs/` も同じ変更で更新する。

## 11. 外部情報

生物学的パラメータを追加するときは一次論文または公式データを優先し、出典を `docs/references.md` または対象文書へ追加する。

## 12. 大きな変更

以下を変更する場合は、理由・代替案・影響を文書化する。

- 神経ダイナミクスモデル
- 可塑性則
- MaleCNS基準版
- 身体モデル
- 感覚変換
- CNS→身体写像
- 報酬・嫌悪刺激対象
- チェックポイント形式

## 13. 判断に迷った場合

「一般的なAIならどうするか」ではなく、まず次を問う。

> 実際のハエに同じ経験を与えるなら、外部から脳へ何が入るか？

その入力を仮想神経系に与え、その後の変化は可能な限り神経系内部の機構に任せる。
