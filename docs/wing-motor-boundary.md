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

目標経路:

```text
MaleCNS全CNS
  -> VNC premotor circuit
  -> 個々のwing motor neuron
  -> 個々の筋または同定可能なmotor unit
  -> 筋活動・力学
  -> wing hinge torque
  -> FlyBody / MuJoCo
```

これは「MNベクトルから行動をdecodeする」処理ではない。神経から筋、筋から関節力学という末梢過程を実装する。

## 2. 現在のMaleCNS inventory

`scripts/data/prepare_wing_motor_map.py` はMaleCNS v1.0で

```text
superclass = vnc_motor
subclass   = wm
```

に該当するreleased neuronをbody ID単位で抽出する。

2026-09-14時点の実行では:

- 67 neurons
- left 33
- right 34
- 26 annotated types

が得られた。

左右集団を平均した値をmotor commandとして使用しない。bridge側では個々のbody IDのspikeを読む。

## 3. 筋標的の来歴

MANCのwing MN同定では、wing muscle systemを大きく以下へ分ける。

### power muscle

- DLM
- DVM1
- DVM2
- DVM3

### direct steering muscle

- b1, b2, b3
- i1, i2
- iii1, iii3, iii4
- hg1, hg2, hg3, hg4

### indirect control / thoracic tension muscle

- tp1, tp2
- tpn
- ps1, ps2
- TTM
- satellite TTM

`prepare_wing_motor_map.py` はこれらの既知対応だけをmetadataとして付与する。

重要な例外:

- `MNwm35 -> iii4` はputative。
- `MNwm36` の筋標的はMANCでも未確定。pleurosternal muscleの可能性は仮説として報告されているが、virtual-flyではidentified扱いしない。
- `tpn` はtp1/tp2の両方への可変的なinnervationが報告されているため、どちらか一方へ固定しない。
- `DLMn c-f`、`DVMn 1a-c`、`DVMn 2a,b`、`DVMn 3a,b` はMaleCNSのtype labelだけでは各body IDを各筋線維へ一意に割り当てられない。追加証拠なしに順番で割り当てない。

## 4. FlyBody側の制約

FlyBodyの公開モデルは解剖学的な26種類の翼筋を直接モデル化していない。

現在のwing hingeは左右それぞれyaw / roll / pitchの3 DoFであり、FlyBody論文の標準flight modelではwing jointへidealized actuatorを作用させる。論文自身もactuator control signalを生物学的な筋活動として解釈しないよう注意している。

したがって、wing MNを6個のwing DoFへ直接重み付き和で変換するだけでは本プロジェクトの目標を満たさない。

## 5. 実装する末梢層

次の順で実装する。

### 5.1 MN spike -> motor-unit state

各body IDを独立状態として保持する。

最低限:

- spike history
- neuromuscular activation state
- decay / refractory-like peripheral dynamics

複数MNを先に平均しない。

### 5.2 motor unit -> muscle activation

筋ごとに生理学的に妥当なactivation dynamicsを実装する。

power musclesとsteering musclesを同じ式で扱わない。DLM/DVMはasynchronous indirect flight muscleであり、steering muscleのphasic/tonic recruitmentとは性質が異なる。

未知のパラメータは `calibrated` として隔離し、実測値として扱わない。

### 5.3 muscle -> wing hinge force/torque

筋の付着・hingeへの作用を物理モデルとしてMuJoCoへ与える。

第一段階で完全な3D muscle wrappingが不可能な場合は、FlyBody論文が将来拡張として述べるvirtual-muscle方式、すなわち筋活動を関節torqueへ写す機械的モデルを採用できる。ただし:

- game stateを見ない
- gap位置を見ない
- rewardを見ない
- CNS集団をactionへdecodeしない
- 各筋の寄与は独立に計算して物理的に合成する

ことを必須とする。

## 6. 強化刺激との分離

運動境界と強化刺激境界は別問題として扱う。

Flyppyで成功・失敗イベントが発生したとき、外部コードがsynaptic weightやscalar rewardを神経ランタイムへ渡してはならない。

許される境界は、文献とMaleCNS annotationから選んだ実在DAN body IDへ外部電流を注入することだけである。その後のdopaminergic activityとplasticityはCNS内部のreleased connectivity上で発生させる。

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
- `literature`: wing MNと筋標的の同定、power/direct-steering/indirect-control分類。
- `putative`: MNwm35 -> iii4。
- `unknown`: MNwm36の筋標的、untyped released wing MN、grouped type内の個別筋線維対応。
- `calibrated`: 将来導入する、実測が不足する筋activation parameter・virtual muscle torque parameter。
