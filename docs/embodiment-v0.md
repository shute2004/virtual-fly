# Embodiment v0

## 1. 現在の範囲

このsliceでは、外部policyを導入せず、MaleCNS神経ランタイムをFlyBodyとFlyppy環境へ接続する。

実装済み:

- FlyGym 2.1.0 / FlyBodyによる身体・MuJoCo物理。
- FlyGym 2.1.0のFlyBody変換で省略されている左右wing fluid geometryの復元。
- 元FlyBody飛翔タスクに合わせたwing actuator gain、wing stiffness / damping、50 µs physics timestep、空気密度・粘性の単位補正。
- FlyBodyの左右wing DoF駆動。
- FlyBody公式飛翔条件に基づく47.5度body pitchと、episode開始時だけ与える一回限りの前進初速度。
- JSONL stdin/stdoutで常駐するRust MaleCNS bridge。
- body IDを指定した個々のMaleCNSニューロンへの直接外部電流刺激。
- MaleCNS注釈からのbilateral DNg02 / PAM08 / PPL1抽出。
- MaleCNS L1の `assignedOlHex1` / `assignedOlHex2` に基づくretinotopic lamina-cartridge map。
- 各L1へ実際に接続するR1-R6 photoreceptor body IDの解決。
- FlyBody眼カメラの局所受光量を、対応するR1-R6だけへ電流として与える感覚境界。
- DNg02左右集団活動から左右wing-beat amplitudeへのmotor adapter。
- 上下ゲート、床、天井、明示的MuJoCo contact pairを持つFlyppy world。
- ゲート通過時のPAM08候補刺激、衝突時のPPL1候補刺激。
- 上記を一本化した閉ループ局所可塑性学習runner。
- 全CNS動的・可塑性状態のcheckpoint保存と復元。
- 任意の3Dライブ表示とepisodeごとのMP4保存。
- 任意の低頻度シナプス変化snapshot。

過去の試作 `visual_motion_encoder.py` は、外部でT4/T5相当の運動特徴を計算していたため、現在のFlyppy学習経路では使用しない。

## 2. 情報経路

```text
Flyppy物理環境
  -> FlyBody raw eye cameras
  -> MaleCNS L1 lamina cartridgeごとの局所受光
  -> そのL1へ実接続するR1-R6 body IDへの外部電流
  -> whole MaleCNS + local plasticity
  -> bilateral DNg02
  -> wing-beat amplitude adapter
  -> FlyBody / MuJoCo
  -> 位置・衝突・ゲート通過
  -> PAM08またはPPL1刺激
  -> local plasticity
```

ゲーム側は障害物座標や「上へ行け」「下へ行け」という命令をCNSへ渡さない。環境がsynaptic weightを直接変更することもない。

重要なのは、CNSへ入る前に外部コードが運動方向・edge・障害物・gap位置を計算しないことである。視覚上の意味情報はMaleCNS内部の実接続と神経状態の時間発展から生じさせる。

## 3. 視覚入力

### 3.1 実測情報とneural superposition

MaleCNS公式annotationにはoptic-lobeのhex column座標 `assignedOlHex1` / `assignedOlHex2` が含まれる。

`scripts/data/prepare_retinotopic_vision.py` は、column座標を持つ実際のL1ニューロンをlamina cartridgeの空間単位とし、そのL1へreleased MaleCNS connectivity上で実際にpresynaptic connectionを持つannotated R1-R6ニューロンだけを同じoptical columnへ割り当てる。

これはショウジョウバエのneural superposition――同じvisual axisを見る近傍ommatidia由来のR1-R6が同じlamina cartridgeへ収束する――を利用する。R1-R6のbody ID順やommatidiumの配列順からcolumn所属を推測しない。

解決結果は:

```text
artifacts/malecns-v1.0/retinotopic-vision-v1.json
```

へ保存する。十分なcolumn数を解決できない、同じR1-R6が複数columnへ割り当たる、同じL1 column座標が重複する、といった場合は、適当な順序対応や全体平均へfallbackせず実行を失敗させる。

### 3.2 感覚変換境界

`scripts/embodiment/malecns_retina.py` は各眼についてMaleCNS hex latticeをFlyBodyのraw eye cameraへ展開し、それぞれのcolumn位置の局所受光値だけを読む。

その局所値を、当該L1 cartridgeへ実際に接続しているR1-R6への外部電流へ変換する。

```text
one observed L1 cartridge / optic column
  -> one local eye-camera sample
  -> local photoreceptor transduction scale
  -> its observed presynaptic R1-R6 currents
```

column間の平均、pooling、Reichardt-like motion detector、edge detector、object detectorは使用しない。T4/T5を含む下流視覚ニューロンの応答はMaleCNS自身に計算させる。

### 3.3 provenance

- `observed`: MaleCNS body ID、released R1-R6 -> L1 connectivity、L1の `assignedOlHex1` / `assignedOlHex2`。
- `literature`: neural superpositionにより、同じoptical axisのR1-R6が同じlamina cartridgeへ収束するという配線原理。
- `calibrated`: MaleCNS hex latticeからFlyBody eye-camera平面への幾何投影。
- `calibrated`: 局所受光値からR1-R6へ注入するcurrentのscale。

後二者は今後より生理学的な光学系・phototransductionモデルへ置換可能だが、置換時も局所性とretinotopyを壊さない。

## 4. 運動出力と飛翔物理

DNg02は最初の粗い飛翔出力として使用する。DNg02集団活動はwing stroke amplitude / thrust regulationと関連するため、左右DNg02活動を左右wing-beat amplitudeへ写す。

現在のanalytic wing beatは、完全なmotor-neuron -> flight-muscleモデルではない。下位の飛翔運動生成機構を暫定的にまとめたadapterである。adapterはFlyppyの障害物位置や報酬状態を見ない。

FlyGym 2.1.0の実験的FlyBody統合では、元FlyBody XMLにある `wing_left_fluid` / `wing_right_fluid` geometryが変換時に省略されている。そのままでは元FlyBody飛翔タスクと同じ空力条件にならないため、`scripts/embodiment/flybody_flight_physics.py` で以下を復元する。

```text
physics timestep     5e-5 s
body pitch           47.5 deg
wing position gain   source 18 -> FlyGym mm系 1800
wing stiffness       source 0.01 -> 1.0
wing damping         source 0.007769230 -> 0.776923
fluid coefficients   [1.0, 0.5, 1.5, 1.7, 1.0]
air density          source 0.00128 -> FlyGym mm系 1.28e-6
air viscosity        source 0.000185 -> FlyGym mm系 1.85e-5
```

元FlyBodyはcm、FlyGym版はmmを使うため、長さはx10、torque-like量はx100、densityはx1e-3、viscosityはx1e-1として変換する。

Flyppyではepisode開始時にのみ +X 方向の初速度を与える。既定値は `300 mm/s` で、元FlyBody vision-flight taskが使う20–40 cm/sの中央である。これは継続的な外部policyではなく飛翔開始条件であり、その後の並進速度を外部から維持・補正しない。

```text
episode reset
  -> 47.5 deg flight pose
  -> vx = 300 mm/s を一度だけ設定
  -> 以後はMuJoCo物理 + CNS由来wing controlのみ
```

`flybody_flight_envelope.py` は同一初速度でwing fluidあり/なしの対照を取り、初速だけで前進したケースを空力成立と誤認しないようにする。

## 5. 学習とcheckpoint

実行中、同一のMaleCNS runtimeをepisode間で維持するため、可塑的シナプスと神経活動状態は前episodeの経験を保持する。

結果イベントは以下の神経刺激へ変換する。

```text
gate pass -> PAM08候補群を刺激
collision -> PPL1候補群を刺激
```

外部optimizer、backpropagation、Q-learning、policy gradient、scalar rewardによるweight直接更新は使わない。

checkpointは既定で以下へ保存する。

```text
artifacts/experiments/flyppy-v0/checkpoint/
├── manifest.json
├── membrane.f32le
├── spikes.u32le
├── refractory.u32le
├── activity-trace.f32le
├── modulation.f32le
├── weights.f32le
└── eligibility.f32le
```

保存対象は膜電位、現在のspike状態、refractory counter、activity trace、neuromodulation状態、全synaptic weight、全eligibility traceである。

connectome topology、neurotransmitter annotation、数値モデルparameter、PAM/PPL1等のmodulator roleは同じsnapshot/configurationから再構成する。FlyBody・コース状態はepisode境界でリセットするためcheckpointには含めない。

## 6. 実行

接続・物理確認:

```bash
bash scripts/dev/embodiment.sh
```

この確認には、MaleCNS retinotopic R1-R6 mapの解決、raw eyeからの局所photoreceptor current、復元したflight fluid geometry・空気parameter・wing gain/stiffness/damping、free-flight motor effect、flight envelopeを含む。

通常のヘッドレス学習:

```bash
bash scripts/dev/train_flyppy.sh
```

長く回す例:

```bash
bash scripts/dev/train_flyppy.sh --episodes 100
```

既存checkpointから継続する場合:

```bash
bash scripts/dev/train_flyppy.sh \
  --resume-checkpoint artifacts/experiments/flyppy-v0/checkpoint \
  --episodes 100
```

感覚境界の設計を変更したcheckpointを混在させないこと。旧T4/T5外部encoderで生成したcheckpointは、retinotopic R1-R6版の学習継続用として扱わない。

## 7. 可視化

3Dライブ表示:

```bash
bash scripts/dev/train_flyppy.sh --render
```

動画保存:

```bash
bash scripts/dev/train_flyppy.sh --record-video artifacts/experiments/flyppy-v0/video
```

シナプス変化snapshot:

```bash
bash scripts/dev/train_flyppy.sh --synapse-trace
```

表示やログのためにretinal current等を集計することはあるが、その集計値をCNS入力へ戻してはいけない。

## 8. 次の改善点

- MaleCNS optic-column座標とFlyBody眼カメラの光学的対応を、実際のommatidial optical axisに基づいて校正する。
- R1-R6のphototransductionを、単純current scaleから文献ベースの局所生理モデルへ置換する。
- flight motor neuron / flight muscle単位の詳細neuromuscular model。
- checkpointのsnapshot/configuration/sensory-interface fingerprint固定と世代管理。
- 視覚・運動・可塑性parameterの生理学的校正。
- 学習前後比較、対照群、複数seedでの統計評価。
- MaleCNSの実3D neuron morphology / synapse coordinatesを使った解剖学的ビューア。
