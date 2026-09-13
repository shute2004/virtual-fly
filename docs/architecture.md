# システム設計

## 1. 全体像

`virtual-fly` は、神経系・身体・環境・実験制御を明確に分離する。

```text
┌──────────────────────────────────────────────┐
│ Experiment Runner                            │
│ conditions / seeds / checkpoints / logging   │
└──────────────┬───────────────────────────────┘
               │
               v
┌──────────────────────┐      ┌──────────────────────┐
│ Environment          │<---->│ Body / Physics       │
│ Flyppy world         │      │ FlyBody / MuJoCo     │
└──────────┬───────────┘      └──────────┬───────────┘
           │ sensory scene               │ motor state
           v                             ^
┌──────────────────────┐      ┌──────────┴───────────┐
│ Sensory Transduction │      │ Motor Interface      │
│ visual / other       │      │ CNS -> body mapping │
└──────────┬───────────┘      └──────────^───────────┘
           │                              │
           v                              │
┌──────────────────────────────────────────────┐
│ Neural Runtime                               │
│ adult male CNS                               │
│ neural dynamics / synapses / plasticity      │
│ neuromodulation / structural change          │
└──────────────────────────────────────────────┘
```

学習アルゴリズムを別モジュールとして置かない。学習は `Neural Runtime` 内の局所的な可塑性としてのみ発生する。

## 2. 技術方針

### 2.1 Rust: 神経系コア

Rustを次の用途に使用する。

- 大規模疎グラフの保持
- ニューロン状態の更新
- シナプス伝播
- 可塑性状態の更新
- 構造可塑性
- 神経修飾状態
- チェックポイントの高速入出力
- 将来のWASM対応

Pythonオブジェクトをニューロン・シナプス単位で大量生成する設計は避ける。

### 2.2 Python: 科学実験・統合

Pythonを次の用途に使用する。

- MaleCNS / neuPrint データ取得と前処理
- FlyGym / FlyBody / MuJoCo との接続
- 実験設定
- 可視化・解析
- RustコアへのFFI
- 参照実装・小規模検証

Rust-Python境界は `PyO3` / `maturin` を第一候補とする。

### 2.3 Web: 後段

最終公開段階では神経系コアの一部または全部をWASMへ展開し、可能な部分をWebGPUへ移す。

ただしブラウザ対応を初期実装の制約にしすぎない。まずネイティブ実行で科学的な閉ループを成立させる。

## 3. 推奨ディレクトリ構成

```text
virtual-fly/
├── README.md
├── AGENTS.md
├── Cargo.toml                 # Rust workspace
├── pyproject.toml             # Python package / tooling
├── crates/
│   ├── vf-core/               # 時間・状態・イベント等の共通型
│   ├── vf-connectome/         # MaleCNS内部表現とI/O
│   ├── vf-neural/             # 神経ダイナミクス
│   ├── vf-plasticity/         # 可塑性・神経修飾
│   ├── vf-checkpoint/         # 状態保存
│   └── vf-python/             # PyO3 bindings
├── python/
│   └── virtual_fly/
│       ├── data/              # 取得・変換
│       ├── body/              # FlyBody/FlyGym adapter
│       ├── sensory/           # 感覚変換
│       ├── motor/             # 運動系写像
│       ├── envs/              # Flyppy等
│       ├── experiments/       # 実験runner
│       └── analysis/          # 解析
├── configs/
│   ├── neural/
│   ├── plasticity/
│   ├── body/
│   └── experiments/
├── scripts/
│   ├── data/
│   └── dev/
├── tests/
├── docs/
└── artifacts/                 # gitignore対象。実験生成物
```

これは目標構造であり、最初から空ディレクトリを大量に作る必要はない。

## 4. 神経系内部表現

### 4.1 NeuronTable

ニューロン単位の静的属性と動的状態を分ける。

静的属性例:

- `body_id`
- `cell_type`
- `superclass`
- `region`
- `neurotransmitter`
- データ来歴

動的状態例:

- 膜電位または採用モデル固有の状態
- 活動状態
- refractory state
- 神経修飾状態
- モデル固有内部変数

実装ではStructure of Arraysを優先し、全個体を連続配列で保持できるようにする。

### 4.2 SynapseStore

MaleCNSには非常に多数のシナプスがあるため、疎構造を前提とする。

初期候補:

- CSR / CSC相当の静的ベース接続
- 可塑性値の別配列
- 構造変更用delta layer

構造可塑性を導入した後も、毎ステップ巨大なCSR全体を再構築しない。

概念的には:

```text
base connectome (immutable source snapshot)
        +
plastic state (mutable)
        +
structural delta (created / removed edges)
        =
current functional connectome
```

とする。

実測コネクトームそのものを破壊的変更せず、「現在の仮想個体の状態」を別レイヤとして保持する。

## 5. シミュレーション時間

身体物理と神経系では必要な時間刻みが異なる可能性が高い。

```text
physics step
  ├─ neural substep 1
  ├─ neural substep 2
  ├─ ...
  └─ neural substep N
```

の多重時間刻みを許容する。

固定値は設計段階で決めず、採用する神経モデルとFlyBody側の安定条件から決める。

## 6. イベントと刺激

外部環境から神経系への入力は、抽象的な学習命令ではなく `StimulusEvent` として表す。

例:

```text
StimulusEvent {
    target: neuron set / sensory channel,
    waveform: ...,
    onset: ...,
    duration: ...,
    provenance: visual | reward-circuit | aversive-circuit | ...
}
```

重要なのは、`reward = +1` を可塑性エンジンへ渡さないことである。

成功イベントを受けた実験層が、対応する神経回路へ与える刺激波形へ変換する。

## 7. 可塑性アーキテクチャ

可塑性はプラガブルにするが、一般の機械学習optimizer形式にはしない。

```text
neural activity
+ local synapse state
+ neuromodulator state
+ time
        ↓
local plasticity rule
        ↓
synapse state change
```

構造可塑性も別インターフェースとして持たせる。

```text
activity history / local state
        ↓
formation / pruning candidate
        ↓
structural delta
```

初期段階では、機能可塑性を先に検証し、構造可塑性は後から追加してよい。

## 8. 感覚系

### 8.1 視覚

Flappy環境では視覚を最優先する。

処理境界:

```text
MuJoCo scene
   ↓
compound-eye renderer
   ↓
photoreceptor / peripheral visual model
   ↓
MaleCNS visual entry points
```

MaleCNSデータに含まれない末梢感覚器は補完モデルとして明示する。補完モデルをCNSの実測コネクトームと混同しない。

### 8.2 自己受容・機械感覚

FlyBodyから得られる関節角、角速度、接触、力などを、生物学的感覚入力へ変換する層を後から追加する。

## 9. 運動系

最初の運動変換は、下降・運動系ニューロン活動からFlyBodyアクチュエータへの写像とする。

```text
CNS output
   ↓
biological mapping adapter
   ↓
body actuation
```

このadapterには環境認識・方策・ゲーム攻略ロジックを入れない。

将来的には:

```text
motor neuron
  ↓
neuromuscular junction
  ↓
muscle model
  ↓
body joint / wing
```

へ置き換える。

## 10. チェックポイント

チェックポイントは「重みファイル」ではなく仮想個体の状態である。

推奨構造:

```text
manifest.json
neurons.bin
synapses.bin
plasticity.bin
structural_delta.bin
neuromodulation.bin
body_state.bin
rng.bin
```

実際の形式はベンチマーク後に決める。大規模配列の保存には、圧縮可能で部分読み出し可能な形式を優先する。

## 11. 実験IDと個体系譜

同じ初期コネクトームから複数個体を分岐させるため、個体に親子関係を持たせる。

```text
male-cns:v1.0
      ↓
fly/base-0000
      ├── fly/0001
      ├── fly/0002
      └── fly/0003
             └── fly/0104
```

各派生個体について、親チェックポイント、経験履歴、コード版を追跡する。

## 12. ブラウザ分散実験への境界

将来、ブラウザ側へ渡す単位は「任意コード」ではなく、署名・バージョン付きの実験パッケージとする。

クライアントから返る結果は信頼しない。

- 入力条件を固定した再検証
- ハッシュ・バージョン確認
- 異常値検出
- 重複排除
- 一部実験の複数クライアント再実行

を想定する。

## 13. 最初に実装しないもの

- ブラウザ分散計算
- 巨大なWeb UI
- 全感覚器
- 完全な筋肉モデル
- 全種類の可塑性
- 発生過程

まずは小規模神経サブグラフで「動く・変わる・保存できる」を確認し、その後に全CNSと身体を接続する。
