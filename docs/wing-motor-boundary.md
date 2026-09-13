# 翼運動ニューロンと末梢筋境界

## 1. 目的

Flyppy学習でCNS外に行動決定器を置かないため、MaleCNSから身体への境界を個々のreleased wing motor neuronまで下げる。

禁止する経路:

```text
DNg02集団活動
  -> 平均発火率
  -> 左右差・平均
  -> wing amplitude
```

現在の学習経路:

```text
FlyBody eye
  -> R1-R6 body IDへの局所電流
  -> MaleCNS全CNS
  -> VNC premotor circuit
  -> 個々のwing motor neuronのspike
  -> 個々のmotor-unit / neuromuscular state
  -> 同定済みwing muscle
  -> virtual-muscle torque
  -> FlyBody / MuJoCo
```

これはMNベクトルから行動をdecodeする処理ではない。神経から筋、筋から関節力学という末梢過程である。

## 2. MaleCNS inventoryと学習対象

`scripts/data/prepare_wing_motor_map.py` はMaleCNS v1.0で

```text
superclass = vnc_motor
subclass   = wm
```

に該当するreleased neuronをbody ID単位で抽出する。

2026-09-14の実データでは:

- 67 neurons
- left 33 / right 34
- 26 annotated types
- identified 38
- identified_group 22
- identified_variable 2
- putative 2
- unknown 3

初期学習境界では `identified` と `identified_group` の60個だけを使用する。残り7個は推測接続しない。

除外対象の代表例:

- `MNwm35 -> iii4`: putative
- `MNwm36`: muscle target unresolved
- `tpn`: tp1/tp2への可変innervation
- untyped released wing MN

DLM/DVMのgrouped type内でも、各body IDを各筋線維へ追加証拠なしに順番で割り当てない。

## 3. 実装済み末梢層

### 3.1 個別MN spike -> motor-unit state

`scripts/embodiment/wing_muscle_periphery.py` が実装する。

- bridgeから60 body IDのspikeを個別に要求する
- body IDごとに独立したactivation stateを保持する
- power / direct steering / indirect controlで別の時定数を持つ
- bridgeが1個でも要求body IDを返さなければfail closedする
- population firing rateや左右平均を作らない

複数motor unitが同じ筋群へ入る場合は、各unitを独立更新した後、筋での物理的recruitmentとして飽和合成する。これはaction decoderではなく末梢での力発生の収束である。

現在の時定数・spike recruitment量は `calibrated` bootstrap parameterであり、実測MaleCNS定数とは扱わない。

### 3.2 asynchronous power muscle

DLM/DVMはasynchronous indirect flight muscleであり、MNが約200 Hzの各wingbeatを直接タイミングするモデルにはしない。

`scripts/embodiment/flybody_muscle_adapter.py` は、DLM/DVMの低周波activationをthoracic flight oscillatorの利用可能なpowerへ変換する。wingbeat phaseは末梢のstretch activation / thoracic resonanceを近似する機械的状態であり、CNSが選ぶactionではない。

初期episodeでは既に飛行中という条件を表現するため、power muscleのみ初期activationを持つ。これは初期forward velocityやinitial wing phaseと同じepisode initial conditionであり、stepごとの外部motor commandではない。

### 3.3 steering muscle -> torque

FlyBodyには解剖学的wing muscleそのものが存在しないため、各筋を直接MJCF muscleとして接続することは現状できない。そのため第一段階では `qfrc_applied` を使うvirtual-muscle layerを置く。

現時点でtorqueへ作用させるdirect steering muscleは、定性的な作用符号を比較的強く制約できるものだけである。

- b1: stroke-amplitude側
- b2: stroke-amplitude側
- b3: basalar pairへのantagonist側
- i1: stroke-amplitudeを減少させる側

数値gainは実測moment armではなく `calibrated` とする。

その他の同定済みsteering / indirect-control muscleもactivation stateまでは保持するが、作用方向を推測してtorqueを生やさない。

### 3.4 FlyBodyへの接続

旧DNg02 adapterはlegacy physics diagnosticとして残るが、Flyppy learning pathからは外した。

`FlyBodyMuscleAdapter` は継承元の6個のidealized POSITION actuatorを各physics stepで現在角へneutralizeし、旧analytic position commandによる行動を発生させない。その上でvirtual-muscle torqueをwing DOFへ加える。

## 4. 強化刺激

強化刺激とmotor boundaryは別である。

外部コードが神経ランタイムへ渡してよいのは、実在DAN body IDへの電流だけである。

現在のFlyppy実験対象:

```text
gate pass
  -> released PAM01 (PAM-gamma5) DANへcurrent

collision
  -> released PPL101 (PPL1-gamma1pedc) DANへcurrent
```

`reward=+1`、`punishment=-1`、scalar reward、直接weight updateは存在しない。刺激後のdopamineとplasticityはMaleCNS connectivity上のneural runtimeで発生する。

PAM01内にも生理学的heterogeneityがあり得るため、「PAM01の全細胞が自然条件で完全に同一のreward信号」という主張はしない。この刺激対象は明示的な実験条件である。

## 5. 学習起動

`scripts/dev/train_flyppy.sh` は起動前に以下を再生成・検査する。

1. reinforcement / diagnostic groups
2. individual wing-MN map
3. R1-R6 retinotopic map
4. wing neuromuscular boundary smoke test
5. Flyppy closed-loop learning

DNg02 population-average guardは、学習経路が個別MN境界へ移行したため削除した。

## 6. 残る限界

現在の境界は「実MaleCNS -> 個別MN -> 解剖学的筋identity」まではデータ/文献に基づくが、「筋 -> FlyBody hinge」はvirtual-muscle近似である。

今後より高い生体忠実度を得るには、筋ごとの付着点・moment arm・activation/force特性を導入し、FlyBody内に解剖学的muscle modelを追加する必要がある。ただし不明な値を行動最適化のために逆算して埋めない。

## 7. 参考資料

- Lesser et al., *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*, eLife, version of record 2026.
  - https://elifesciences.org/articles/96084
- Ehrhardt et al., *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/
- Azevedo et al., *Synaptic architecture of leg and wing premotor control networks in Drosophila*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC10312524/
- *How tp1, an indirect wing steering muscle, stabilizes Drosophila's flight*.
  - https://pmc.ncbi.nlm.nih.gov/articles/PMC12637562/
- Vaxenburg et al., *Whole-body physics simulation of fruit fly locomotion*, Nature 2025.
  - https://www.nature.com/articles/s41586-025-09029-4

## 8. provenance

- `observed`: MaleCNS body ID、type、side、released CNS connectivity。
- `literature`: wing MNと筋標的、power/direct-steering/indirect-control分類、DLM/DVM asynchronous flight、b1/b2/b3/i1の定性的作用。
- `putative`: MNwm35 -> iii4。
- `unknown`: MNwm36、untyped MN、grouped type内の個別筋線維対応。
- `calibrated`: peripheral activation time constants、spike recruitment、thoracic oscillator coupling、virtual-muscle gain、torque limit。
