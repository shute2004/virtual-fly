# 現在のarchitecture

[English](architecture.md) · [日本語](architecture.ja.md) · [简体中文](architecture.zh-CN.md)

この文書は、現在のproduction / canonical pathで**実際に実装されているarchitecture**を説明します。roadmapではなく、historical artifactへ現在の意味論を遡及適用するものでもありません。

## 1. System boundary

`virtual-fly` は責務を大きく4つに分けます。

1. **神経系** — MaleCNS dynamics、modulation、eligibility、局所可塑性
2. **感覚・末梢embodiment** — 局所的な物理感覚変換、motor neuron→muscle state
3. **身体・環境** — FlyBody / MuJoCoとFlyppy geometry
4. **実験orchestration** — reset、curriculum条件、persistence、provenance、report

実験層は「物理課題イベントがいつ起きたか」と「そのイベントでどの神経群を刺激するか」を決められますが、target actionや望ましいsynaptic weightを計算しません。

```text
physical scene
    ↓
local sensory transduction
    ↓
MaleCNS neural runtime
    ↓
individual motor-neuron activity
    ↓
peripheral muscle state
    ↓
FlyBody / MuJoCo
    ↓
physical scene
```

## 2. Canonical closed loop

現行canonical Flyppy path:

```text
Flyppy v7 physical environment
        ↓
FlyBody compound-eye geometry
        ↓
local direct-ray sampling (K=13)
        ↓
released MaleCNS R1-R6 body-ID currents
        ↓
whole-MaleCNS signed neural dynamics
        ↓
local eligibility + dopaminergic modulation
        ↓
individual released motor-neuron spikes
        ↓
WholeBodyPeriphery
        ↓
FlyBody v7 physical actuation
        ↓
MuJoCo state transition
        ↓
Flyppy event detection
        ├─ gate passage → selected reward-associated DAN current
        └─ collision    → selected aversive-associated DAN current
```

この経路に外部ニューラルネットワークやaction policyはありません。

## 3. MaleCNS sourceとmutable state

公開MaleCNS dataはmutable training fileではなく、初期構造snapshotとして扱います。

```text
immutable source snapshot
        +
mutable functional weight state
        +
short-term neural/plasticity state
        =
current simulated nervous system
```

Canonical v1:

- 166,700 neurons
- 25,582,938 directed connection edges
- released neurotransmitter / annotation information
- class-DAN = dopamine consensusかつ公開annotation `class=DAN`

学習でsource file自体を書き換えることはありません。

## 4. Neural runtime

大規模神経runtimeは`crates/`以下のRust実装です。Pythonはorchestrationと身体統合を担当し、Rust processがwhole-CNS state evolutionを担当します。

現在のactivity semanticsは、silent、depolarizing activity event、hyperpolarizing activity deviationを区別します。抑制・graded pathwayを単一のunsigned spike bitへ潰さないためです。

主な責務:

- sparse MaleCNS graph保持
- state propagation
- modulation state
- eligibility dynamics
- local weight update
- shared-weight population transaction
- checkpoint load/save

GPUは計算backendであり、別のlearned controllerではありません。

## 5. 局所可塑性

production plasticityは局所則です。概念的にはedge updateが局所eligibilityとpostsynaptic neuromodulationに依存します。

```text
pre/post activity history
        ↓
eligibility
        +
local dopaminergic modulation
        ↓
local bounded synaptic update
```

環境から`reward = +1/-1`、Q値、target action、policy loss等をweight optimizerへ渡す経路はありません。

PlasticFastGraph高速化は走査・更新対象edgeを減らすもので、別のlearning ruleへ置き換えるものではありません。

## 6. Neuromodulatory event

課題イベントは神経刺激へ変換します。

Canonical class-DAN semanticsは公開された実在DAN body IDを使います。reward-associated / aversive-associated subsetは**実験刺激条件**であり、そのlabelに含まれる全neuronがあらゆる自然条件で普遍的なscalar rewardを表すという主張ではありません。

```text
gate passage
    ↓
current into selected reward-associated DAN subset

collision
    ↓
current into selected aversive-associated DAN subset
```

Canonical frozen evaluationでは両方のtask-triggered DAN currentを0にします。

## 7. Vision boundary

production visionはCNS外でobstacle classificationを行いません。

```text
MuJoCo / FlyBody eye geometry
        ↓
per-ommatidium local direct rays
        ↓
local photoreceptor transduction/adaptation
        ↓
corresponding released R1-R6 body IDs
        ↓
MaleCNS visual circuitry
```

CNSへ入るまで空間identityを保持します。外部edge detection、object recognition、motion label、global spatial poolingを「答え」として注入しません。

## 8. Haltere boundary

haltere pathは物理身体から局所mechanical signalを作り、MaleCNS sensory afferentへ写像します。

Canonical v1は推定された97-neuron timing subset（left 46 / right 51）、current gain 0.05、`interaction-load-v2`を使います。

これは明示的に`inferred`なsensory boundaryであり、完全なhaltere physiologyを直接観測したものとして扱いません。

## 9. Motor boundary

production behaviorはDNg02 population-average decoderから生成されません。

```text
released motor-neuron body ID activity
        ↓
WholeBodyPeriphery
        ↓
individual motor-unit / muscle activation state
        ↓
versioned FlyBody adapter
        ↓
physical force/torque in MuJoCo
```

peripheral/body seamには工学的近似がありますが、障害物位置、reward state、desired actionは見ません。

旧DNg02 aggregate経路はlegacy / diagnosticとしてのみ残ります。

## 10. Body version lineage

body version番号は単純な直線継承ではなくhistorical labelです。

```text
v3
└─ v4
   ├─ v5
   ├─ v6
   └─ v7

v8 = v6 measured steering + v7 neutral trim
```

したがって、**v7は「v6に改良を足したもの」ではありません**。Canonical v1がv7を使うのはcanonical条件として固定されたためで、v5/v6の全mechanicsを内包しているからではありません。

## 11. Environment v7

Flyppy environment versionも明示的な実験条件です。v7には物理side wallがあり、3D bodyがgate横を回り込んでpass扱いになる経路を防ぎます。

pass判定はthorax中心を点として扱うだけでなく、wall plane周辺でfull-body geometryを考慮します。

## 12. Population training

production population trainerは1つのglobal weight stateを共有します。

各slotは:

1. 特定global weight versionからepisode開始
2. episode-local plasticity operationを蓄積
3. `async`では独立に終了
4. transactionを最新global weightsへrebase
5. commitして次のglobal weight versionを生成

これは**weight averagingではありません**。

completion orderが変わり得るため、async shared-weight trainingをserial single-fly trainingとbitwise同一とは主張しません。source weight versionとstalenessはprovenanceとして記録します。

## 13. Curriculum boundary

curriculum codeはreset / experience conditionを選びますが、motor actionを選びません。

boundary-band curriculumはfocus/easier experienceとfull-course evaluation logicを分けます。curriculum successはexperiment scheduling stateであり、scalar synaptic reward APIではありません。

## 14. Checkpoint boundary

現在のpopulation persistence semanticsは`global-weights-only-v1`です。

episodeを跨いでlearned global weightsは残しますが、membrane、spike、refractory、activity trace、modulation、eligibility等のepisode-local stateはresetします。

したがって現在のpopulation checkpointを**完全な持続的生物個体状態**と表現してはいけません。

disk上のcheckpoint containerにはcompatibility上の追加配列がありますが、公開上の意味論はraw file layoutではなくweights-only contractに従います。

## 15. Frozen evaluation

公開可能なweight比較では少なくとも次を区別します。

- plasticity OFF
- task-triggered DAN stimulation OFF
- initial / final stored weight state
- 同一body/environment/sensory condition

Canonical v1はこれを満たします。旧fixed evaluationにはevent-triggered DAN currentを停止していないものもあり、canonical evidenceではなくhistoricalとして残す理由の一つです。

## 16. Observer / viewer boundary

3D body viewerとneural viewerはobserver-onlyです。

viewerがいない時、production trainingはviewer snapshotを継続生成する必要がありません。viewerのattach/detachがneural、physical、plasticity、curriculum stateを変えてはいけません。

## 17. 現在のcode ownership

```text
src/virtual_fly/embodiment/   body factory、versioned body seam、retina、haltere、periphery
src/virtual_fly/runtime/      neural bridge、body worker、packed process、telemetry
src/virtual_fly/training/     config、curriculum、population scheduling、checkpoint semantics
src/virtual_fly/playback/     frozen playback/evaluation orchestration
src/virtual_fly/reporting/    stable report/history output
src/virtual_fly/reproducibility.py
                              provenance/hash/semantic validation
src/virtual_fly/semantics.py  compatibility-sensitive semantic identifiers
crates/                       Rust neural runtime / CLI runner
scripts/                      launcher、data preparation、analysis、legacy diagnostic
```

開発者向けmodule mapは [`code-structure.md`](code-structure.md) を参照してください。

## 18. Historical code / artifact

historical script、report、artifactはprovenanceとして有用ですが、存在するだけでcurrent production pathになるわけではありません。

新しいproduction codeはhistorical script module compatibility shimではなく`src/virtual_fly/`を利用します。

historical experiment resultは元のsemantics/provenance labelを保持します。 [`results.ja.md`](results.ja.md) を参照してください。

## 19. 現在canonical biologyとして実装済みとは言えないもの

現在のarchitectureから次を推測してはいけません。

- 全neuron / synapseの完全なconductance-level biophysics
- 完全なreceptor-specific neurotransmitter dynamics
- canonical v1における検証済みstructural synapse formation/pruning
- 完全なperipheral sensory physiology
- 完全なflight-muscle mechanics
- training episodeを跨ぐ完全なpersistent neural state

これらはhidden componentではなく、明示的なmodel boundaryです。
