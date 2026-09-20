# hae実装からvirtual-flyへ取り入れる知見

作成日時: 2026-09-15 08:54 JST

## 1. この文書の目的

外部のショウジョウバエ神経系実装として、以下の2系統を確認した。

1. `hae.satoru.net` / `satorunet/hae`
   - FlyWireコネクトームとShiu et al.系LIFモデルを使い、キノコ体を中心に文字・数字認識を学習させる実装。
2. X上で公開された「ハエ脳へ大脳皮質風の追加回路を接続する」実験
   - 元の166,400ニューロンを残したまま、L2/3・L4・L5・L6のE/I集団からなる166,400ニューロンを追加し、約536万本の人工接続を増設するもの。

この文書の目的は、これらをvirtual-flyへそのまま移植することではない。

`AGENTS.md` の原則、特に以下を維持したまま、開発速度・検証精度・学習の生物学的妥当性を上げるために参考にできる点を整理する。

```text
physical/local stimulus
    ↓
actual sensory pathway
    ↓
whole CNS dynamics
    ↓
actual motor pathway
    ↓
body physics
    ↓
environmental consequence
    ↓
neuromodulatory stimulation
    ↓
local plasticity
```

## 2. 参照先

### 2.1 hae.satoru.net

- X投稿:
  https://x.com/satorunet/status/2099527925104341161
- FAQ:
  https://hae.satoru.net/faq/?lang=en
- 技術説明:
  https://hae.satoru.net/tech/?lang=en
- GitHub:
  https://github.com/satorunet/hae
- 神経モデル説明:
  https://github.com/satorunet/hae/blob/main/flybrain/README.md
- 文字認識・学習実装:
  https://github.com/satorunet/hae/blob/main/juku/reader.mjs

### 2.2 大脳皮質風追加回路の実験

- X投稿:
  https://x.com/WMjjRpISUEt2QZZ/status/2099302337127088456

添付説明では、元の166,400ニューロンを保持したまま、同数の人工ニューロンをL2/3・L4・L5・L6のE/I集団として追加している。

追加シナプスは約5,368,640本で、主に以下からなる。

```text
追加回路内部         5,324,800
既存感覚PN -> L4        9,696
既存MBON -> L2/3        3,104
L6 -> 既存感覚PN       31,040
```

これは実際の人間大脳皮質コネクトームの移植ではなく、Potjans-Diesmann系の層構造・E/I比・接続傾向を参考にした人工回路である。

## 3. 結論

virtual-flyで最も参考価値が高いのは、haeの文字認識アルゴリズムそのものではなく、次の4点である。

1. **タスク投入前に参照実装と厳密比較する検証方法**
2. **可塑性を生物学的に既知の回路へ局所化する考え方**
3. **疎な神経活動を利用したevent-driven実行**
4. **観測済みコネクトームと追加・推定回路を明確に分離するdelta/overlay設計**

優先順位としては 1 > 2 > 3 > 4 と考える。

## 4. 最優先: 各レイヤにgolden/reference testを持つ

haeの神経モデル実装で最も参考になるのは、WebAssembly化したLIFモデルをBrian2参照実装と比較し、決定論的刺激でspike-for-spike一致まで確認している点である。

公開説明では、決定論テストで完全一致、Poisson入力でも発火率相関がほぼ1になるところまで検証している。

virtual-flyでも、今後は最終タスクの成功率を見る前に、各レイヤに独立した「既知入力 -> 既知出力」のテストを持つべきである。

直近のFlyBody飛翔問題はその必要性を強く示している。

診断では最終的に、質量や空力強度ではなく姿勢・座標系の向きが主因と特定された。

```text
本家姿勢 -47.5°:
total_vertical = 1.036 BW

total force magnitude:
1.055 BW
```

つまり、学習を続けても解決しない身体物理の不整合が下流に存在していた。

今後は最低でも以下を独立検証する。

```text
1. retina / sensory transduction
既知の局所光刺激
-> 対応するR1-R6入力

2. neural runtime
既知の小回路・既知刺激
-> CPU参照実装
-> GPU実装
-> 必要なら外部の既知モデル

3. neuromodulation / plasticity
既知のpre/post/DAN activity
-> 期待される局所weight change

4. motor periphery
既知wing MN spike train
-> motor-unit activation
-> muscle activation

5. wing/body physics
既知wing trajectory
-> force / torque / body trajectory

6. full closed loop
上記1〜5が成立した後にのみ課題学習を評価
```

特にFlyBody系では、論文・本家コードで既知のhover trajectoryを入力した場合に、平均鉛直力が体重と整合することを自動テストにする価値が高い。

### 実装方針

- 最終タスク用テストと物理・神経単体テストを分離する。
- golden testは学習checkpointを必要としない形を優先する。
- CPU/GPU backend間の一致も同じ入力で比較する。
- 速度最適化前に参照実装との一致を固定する。

これは `AGENTS.md` の「最適化より検証を優先する」と一致する。

## 5. 高優先: 可塑性を回路・細胞型ごとに分ける

haeでは、全シナプスへ同じ学習則を適用せず、主にKC -> MBONのような、ショウジョウバエで可塑性がよく研究されている部位へ学習を集中させている。

virtual-flyの現行bootstrap可塑性は、一般化された局所三因子則を広く適用できる利点がある一方、実際のハエにおける回路ごとの差をまだ十分に表現していない。

将来は、単一のglobal plasticity ruleではなく、例えば以下のようなregistry形式を検討する。

```text
synapse / circuit class
    ↓
plasticity model
    ↓
parameter provenance
```

概念例:

```text
KC -> MBON
    -> mushroom-body-specific plasticity

visual circuit
    -> known visual adaptation/plasticity rule, if supported

flight-control circuit
    -> literature-supported rule, if available

unknown connection
    -> fixed, or clearly labelled bootstrap rule
```

重要なのは、学習結果を良くするために恣意的に可塑性を追加するのではなく、文献・実測根拠の有無で分けることである。

これにより、

- credit assignmentの探索空間縮小
- 不要な全CNS weight driftの抑制
- 学習結果の解釈性向上
- 生物学的妥当性向上

が期待できる。

## 6. 高優先: DAN / compartment単位の神経修飾を調べる

haeのキノコ体学習では、DANとMBON compartmentの対応を明示的に扱っている。

virtual-flyでも、現在の粗いpostsynaptic modulationを将来精密化する際は、単純な「報酬DAN」「嫌悪DAN」だけで終わらせず、

```text
DAN identity
projection / compartment
postsynaptic target
local plasticity eligibility
```

を可能な範囲で実データ・文献へ寄せる価値が高い。

ただし、haeの以下の部分はそのまま採用しない。

```text
external truth label
-> dopamine(group, +1)
-> dopamine(wrong_group, -1)
```

virtual-flyでは外部コードが正解actionや誤答classを知る設計にしない。

原則は維持する。

```text
environmental event
-> actual/relevant DAN stimulation
-> released CNS connectivity
-> local modulation
-> local plasticity
```

特に、外部から負のdopamine値を直接注入するような設計はcanonical pathでは避ける。

## 7. 中〜高優先: event-driven神経runtimeを調査する

haeのWebAssembly runtimeは、全ニューロンを毎step必ず更新するのではなく、次の入力までにthresholdへ到達し得ないニューロンをスキップするevent-driven方式を使っている。

FlyWire 127k規模でも、典型的には活動ニューロンだけを処理することで高速化している。

virtual-flyでもMaleCNS活動が十分疎い場合は検討価値がある。

ただし、以下の点から「そのまま移植」はしない。

- MaleCNSにはgraded neuronが多い。
- R1-R6などは単純なbinary spike LIFだけでは扱いにくい。
- 現行runtimeはpositive / negative activity deviationを表現している。
- GPUで疎event queueを使うと、場合によってはdense parallel updateより遅くなる。

したがって、まずプロファイルを取る。

検討案:

```text
A. 現行GPU dense runtime
B. CPU event-driven reference prototype
C. activity-frontier sparse GPU prototype
```

同じsnapshot・同じ刺激で結果を比較し、速度だけでなく数値結果の一致を確認する。

event-driven化は神経モデルの意味を変えない「計算最適化」であることを条件とする。

## 8. 中優先: observed connectomeと追加回路をoverlayとして分離する

大脳皮質風追加回路の実験から、その人工皮質自体をcanonical virtual-flyへ持ち込む必要はない。

参考になるのは、元の神経系を保持したまま追加部分を別レイヤとして扱う考え方である。

virtual-flyでは既にsource MaleCNS snapshotをimmutableにし、個体stateを分離する方針がある。

これをさらに明示化して、MaleCNSに含まれない末梢系・推定回路などを次のように管理するとよい。

```text
C_observed
    MaleCNS v1.0 observed graph

Delta_peripheral
    retina / receptor / NMJ etc.

Delta_inferred
    dataから推定した接続

Delta_assumed
    未知部分を埋める最小仮定

Delta_plastic
    個体経験で変化した機能状態
```

追加接続には最低でも以下を持たせる。

```text
source
provenance label
reason
seed if generated
generation algorithm
version
```

同じseedから同じ補完回路を再生成できることも重要である。

## 9. 大脳皮質風追加回路から直接採用しないもの

canonical virtual-flyの目的は「ハエを強いAIへ改造すること」ではなく、「実際のハエ神経系・身体に近い個体を構築すること」である。

したがって、以下は通常経路へ入れない。

### 9.1 166,400人工ニューロンの追加

これは実際のDrosophila anatomyではない。

将来的にchimera / synthetic augmentation実験を行う場合は、canonical flyとは別experimentとして扱う。

```text
canonical virtual fly
!=
synthetic cortex augmented fly
```

checkpoint、manifest、レポートも混在させない。

### 9.2 Potjans-Diesmann cortical microcircuitの直接移植

層構造・E/I比を参考にした人工ネットワークであり、人間皮質そのもののconnectomeではない。

したがって「人間神経系を移植した」と扱わない。

## 10. haeから採用しないもの

haeは非常に興味深いが、目的がvirtual-flyと異なる。

以下はcanonical pathへ入れない。

### 10.1 pixel -> olfactory PN

haeでは画像pixelを685本の嗅覚projection neuronへ人工的に割り当てている。

virtual-flyでは、

```text
3D scene
-> compound eye
-> ommatidium
-> R1-R6
```

を維持する。

### 10.2 MBONを人工クラスへ分割すること

haeではMBON群を数字・ひらがなのclassとして外部から割り当てる。

これは認識器としては合理的だが、virtual-flyでは意味ラベルをCNSへ埋め込まない。

### 10.3 external argmin classifier

haeでは複数MBON compartmentのdriveを比較し、最小値のclassを答えとして選ぶ。

virtual-flyでは外部action decoderを作らない。

### 10.4 scripted IK / hand-made flight

haeの文字を書く身体動作、飛行等の一部は手作り・IK・scripted motionを利用する。

virtual-flyでは、

```text
actual motor neuron
-> muscle
-> physical body
```

を優先する。

### 10.5 whole brainを切って一部回路だけ動かすこと

haeではfull-brain LIFに一部感覚入力を与えるとrunaway activityが発生するため、文字学習では主にmushroom-body周辺以外をsilenceしている。

virtual-flyではこれを通常解決策にしない。

むしろこの現象は、whole-CNS stability testを充実させるための参考事例として使う。

## 11. whole-CNS stability testを増やす

haeでは、特定感覚入力によって全脳LIFが自己持続的な高活動状態へ入ることが確認されている。

virtual-flyでも、現在のone-pulse stability calibrationを拡張する価値がある。

候補:

```text
single pulse
short burst
sustained weak input
sustained strong input
R1-R6 stimulation
mechanosensory stimulation
DAN stimulation
mixed sensory stimulation
```

各条件について、

```text
activity decay
active neuron fraction
max membrane
persistent activity after input off
runaway detection
```

を記録する。

目的は活動を人工的に小さくすることではなく、モデル由来の不安定性と実装バグを区別することである。

## 12. 推奨する実装順

この文書をもとに実装を進める場合、以下の順を推奨する。

### P0: reference/golden tests

最優先。

特に:

```text
FlyBody known hover trajectory -> expected force
CPU neural reference -> GPU equality
known stimulus -> sensory boundary
known MN spikes -> muscle activation
known DAN/pre/post pattern -> plasticity result
```

既存の学習挙動を変えずに追加する。

### P1: plasticity registry設計

まだ全置換しない。

まず現行ruleを`bootstrap_generic`として明示し、将来の回路固有ruleを差し込める構造へする。

例:

```text
PlasticityRule
PlasticitySelector
SynapseClass / CircuitClass
Provenance
```

### P2: mushroom-body-specific plasticityの調査

hae実装を参考にするが、一次論文・公式データを確認してvirtual-fly用に設計する。

haeのclassification-specificな正解ラベル注入は持ち込まない。

### P3: whole-CNS stability matrix

感覚種・入力強度ごとの安定性を自動測定する。

### P4: event-driven prototype

profileで疎性が確認された場合のみ進める。

CPU referenceで意味保存を検証してからGPUへ進む。

### P5: overlay/delta provenance強化

peripheral / inferred / assumed / learnedを機械可読に分離する。

## 13. 実装時の判断基準

新しいアイデアを取り込む際は、毎回以下を確認する。

```text
1. 実際のハエで対応する局所過程があるか
2. 既存の観測データで置き換えられる部分はないか
3. 外部コードが意味・正解・行動を決めていないか
4. source connectomeを汚染していないか
5. 参照実装またはgolden testで単体検証できるか
6. observed / literature / inferred / assumed / calibratedを区別できるか
```

## 14. 要点

今回の外部実装から、virtual-flyへ最も持ち込む価値があるのは「賢くするアルゴリズム」ではない。

持ち込むべきなのは、

- 参照実装との厳密検証
- 回路固有の局所可塑性
- 神経修飾の局在化
- sparse/event-driven計算の可能性
- observed graphと追加deltaの明確な分離

である。

一方、

- 人工class decoder
- pixelを嗅覚系へ入れる変換
- scripted body action
- whole CNSの大部分を停止
- artificial cortexをcanonical flyへ追加

は、virtual-flyの目的とは一致しないため通常経路へは入れない。

短期的には、**P0のgolden/reference testsを最優先する**。

直近の飛翔問題のように、身体・座標系・感覚・神経・筋肉のどこか1層が壊れているだけで、学習結果全体が誤解される可能性がある。各層の物理・神経的不変条件を学習前に機械的に検証できるようにすることが、最も開発時間を短縮する可能性が高い。
