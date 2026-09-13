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
- MaleCNS注釈からのbilateral DNg02 / PAM08 / PPL1抽出。
- MaleCNS注釈からのT4c / T4d / T5c / T5d左右集団抽出。
- FlyGym複眼のper-ommatidium readout。
- 複眼時系列から上下方向のON/OFF motion energyを生成する暫定視覚encoder。
- DNg02左右集団活動から左右wing-beat amplitudeへのmotor adapter。
- 上下ゲート、床、天井、明示的MuJoCo contact pairを持つFlyppy world。
- ゲート通過時のPAM08候補刺激、衝突時のPPL1候補刺激。
- 上記を一本化した閉ループ局所可塑性学習runner。
- 全CNS動的・可塑性状態のcheckpoint保存と復元。
- 任意の3Dライブ表示とepisodeごとのMP4保存。
- 任意の低頻度シナプス変化snapshot。
- 学習中または学習後に開けるブラウザ3D神経ビューア。

## 2. 情報経路

```text
Flyppy物理環境
  -> FlyGym複眼 / ommatidia
  -> 暫定vertical-motion encoder
  -> MaleCNS T4c/T4d/T5c/T5d
  -> whole MaleCNS + local plasticity
  -> bilateral DNg02
  -> wing-beat amplitude adapter
  -> FlyBody / MuJoCo
  -> 位置・衝突・ゲート通過
  -> PAM08またはPPL1刺激
  -> local plasticity
```

ゲーム側は障害物座標や「上へ行け」「下へ行け」という命令をCNSへ渡さない。環境がsynaptic weightを直接変更することもない。

## 3. 視覚入力の現状

FlyGymからは各複眼のommatidium単位の入力を取得する。

ただし、現在のMaleCNS snapshotだけでは「FlyGymのommatidium i がMaleCNSの個体ニューロンjに対応する」というretinotopic mappingを確定できていない。

そのためv0では、ommatidiaの時間変化からReichardt-likeな局所運動energyを計算し、公開注釈で同定したT4/T5集団へpopulation-levelで入力する。

- T4: ON edge系
- T5: OFF edge系
- c: upward motion
- d: downward motion

これは暫定層であり、任意のneuron-order mappingを生物学的対応として扱わない。個体レベルのretinotopic対応が得られた場合は `scripts/embodiment/visual_motion_encoder.py` を置換する。

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

`flybody_flight_envelope.py` は同一初速度でwing fluidあり/なしの対照を取り、初速だけで前進したケースを空力成立と誤認しないようにする。既定では0.12秒の間に最初の8 mmゲート距離へ到達し、thoraxが設定最低高度以上を維持し、終了時も前進速度が正であるbilateral operating pointが存在することを要求する。

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

保存対象は:

- 膜電位
- 現在のspike状態
- refractory counter
- activity trace
- neuromodulation状態
- 全synaptic weight
- 全eligibility trace

である。

connectome topology、neurotransmitter annotation、数値モデルparameter、PAM/PPL1等のmodulator roleは同じsnapshot/configurationから再構成する。FlyBody・コース状態はepisode境界でリセットするためcheckpointには含めない。

全CNS checkpointは大きいため、既定では8 episodeごとと実行終了時に保存する。`--checkpoint-every 1` とすれば毎episode保存できる。

## 6. 実行

接続・物理確認:

```bash
bash scripts/dev/embodiment.sh
```

この確認には、復元したflight fluid geometry・空気parameter・wing gain/stiffness/dampingのcompile後検証、free-flightでのDNg02駆動差、同一初速での空力あり/なし対照を含むflight envelope検証が含まれる。これが通らない状態ではFlyppy学習を開始しない。

通常のヘッドレス学習:

```bash
bash scripts/dev/train_flyppy.sh
```

初速度を変更する場合:

```bash
bash scripts/dev/train_flyppy.sh --initial-forward-speed-mm-s 250
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

同じ `output-dir` へresumeすると `trajectory.jsonl` は追記され、episode番号も既存traceの続きから採番する。

3Dライブ表示付き:

```bash
bash scripts/dev/train_flyppy.sh --render
```

episodeごとの身体動画保存:

```bash
bash scripts/dev/train_flyppy.sh \
  --record-video artifacts/experiments/flyppy-v0/video
```

ライブ表示と録画は同時指定できる。どちらも指定しない場合、observer camera / rendererは生成せず、学習速度を優先する。

神経可視化用のシナプス変化snapshotも保存する場合:

```bash
bash scripts/dev/train_flyppy.sh --synapse-trace
```

`synapse-trace` は毎control stepで2,558万接続を読み戻さない。既定ではepisode終了時だけ全重みを一時dumpし、集計値と変化量上位64接続だけを `synapse-snapshots.jsonl` に残して一時dumpを削除する。

## 7. 3D神経ビューア

学習後に開く場合:

```bash
bash scripts/dev/view_neural.sh
```

別の実験ディレクトリを開く場合:

```bash
bash scripts/dev/view_neural.sh artifacts/experiments/flyppy-v0
```

学習中に見る場合は、1つ目のターミナルで:

```bash
bash scripts/dev/train_flyppy.sh --synapse-trace
```

2つ目のターミナルで:

```bash
bash scripts/dev/view_neural.sh
```

ビューアはlocalhostだけにbindした静的HTTP serverを起動し、`trajectory.jsonl` と `synapse-snapshots.jsonl` を1秒ごとに再取得する。ビューアを閉じても学習プロセスには影響しない。serverを止める場合はビューアを起動したターミナルでCtrl-Cする。

現在の3D表示は以下を示す。

- T4c/T5cの上向き視覚入力。
- T4d/T5dの下向き視覚入力。
- 左右DNg02発火率。
- PAM08/PPL1刺激イベント。
- 変化したシナプス数、平均・最大weight変化。
- weight変化量上位シナプスを3D発光edgeとして表示。

重要: 上位シナプスのノード座標はbody IDから決定論的に生成した模式配置であり、実際のMaleCNS解剖学的位置ではない。実形態・実シナプス座標を取得できた段階で表示層だけ差し替える。

ビューアのThree.jsはビューアを開いた時だけCDNから読み込む。通常のheadless学習には関与しない。

## 8. 次の改善点

- 個々のommatidiumとMaleCNS視覚ニューロンのretinotopic対応。
- flight motor neuron / flight muscle単位の詳細neuromuscular model。
- 元FlyBody飛翔taskとの差を、leg retraction・wing phase initializationを含めてさらに縮める。
- checkpointのsnapshot/configuration fingerprint固定と世代管理。
- 視覚・運動・可塑性parameterの生理学的校正。
- 学習前後比較、対照群、複数seedでの統計評価。
- MaleCNSの実3D neuron morphology / synapse coordinatesを使った解剖学的ビューア。
