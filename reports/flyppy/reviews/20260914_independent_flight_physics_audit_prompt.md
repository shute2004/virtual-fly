# 独立監査依頼: virtual-fly のFlyBody飛行物理が高度維持できない原因

## 目的

GitHubリポジトリ `shute2004/virtual-fly` の現在の `feat/flybody-flyppy-loop` ブランチについて、FlyBody飛行物理の移植がなぜ十分な揚力を生成できていないのかを、既存の仮説に引きずられず独立に監査してください。

これは既存回答の追認ではなく、ゼロベースの原因分析です。コードを実際に読まずに一般論だけで回答しないでください。

## 上流比較対象

必ず必要に応じて以下もGitHub上の実コードを確認してください。

- `TuragaLab/flybody`
- `NeLy-EPFL/flygym`

特にFlyBodyのflight task、wing beat pattern generator、fruitfly XML、wing actuator semantics、wing joint mechanics、fluid ellipsoid、単位系変換を比較してください。

## 現在までに観測された事実

### 1. virtual-flyの自由飛行能力

CNSを外した物理試験で、98条件すべて高度維持不能でした。

```text
vertical_flight samples=98 sustaining=0 climbing=0
best dlm=0.40 dvm=1.00 vx=300.0 dz=-7.041 max_gain=+0.000 end_vz=+56.468 positive_vz_fraction=0.305
```

質量を比較しても改善しませんでした。

```text
native_compiled_mass_mg=0.984621
reference current_v2_1.09mg sustaining=0 climbing=0
reference flybody_published_0.983mg sustaining=0 climbing=0
reference native_compiled sustaining=0 climbing=0
```

したがって、1.09 mgへの質量正規化だけが主因という説明は弱いです。

### 2. 公式 measured wingbeat を使ったreference lift診断

公式 `wing_pattern_fmech.npy` はFigshareから取得済みでshapeは `(500, 3)` です。

最新診断:

```text
reference_lift root_pitch=0.0 leg_pose=neutral drive=position_physics_rate support_ratio=0.090 track_nmae=yaw:0.1884,roll:0.1985,pitch:0.1649 max=0.1985 peak_force=2267.4
reference_lift root_pitch=0.0 leg_pose=neutral drive=position_source_rate support_ratio=0.094 track_nmae=yaw:0.1674,roll:0.1788,pitch:0.1457 max=0.1788 peak_force=2514.4
reference_lift root_pitch=0.0 leg_pose=neutral drive=source_force_rate support_ratio=0.133 track_nmae=yaw:0.1470,roll:0.1755,pitch:0.1464 max=0.1755 peak_force=1800.0
reference_lift root_pitch=0.0 leg_pose=source_retracted drive=position_physics_rate support_ratio=0.090 track_nmae=yaw:0.1884,roll:0.1985,pitch:0.1649 max=0.1985 peak_force=2267.4
reference_lift root_pitch=0.0 leg_pose=source_retracted drive=position_source_rate support_ratio=0.094 track_nmae=yaw:0.1674,roll:0.1788,pitch:0.1457 max=0.1788 peak_force=2514.4
reference_lift root_pitch=0.0 leg_pose=source_retracted drive=source_force_rate support_ratio=0.133 track_nmae=yaw:0.1470,roll:0.1755,pitch:0.1464 max=0.1755 peak_force=1800.0
reference_lift root_pitch=47.5 leg_pose=neutral drive=position_physics_rate support_ratio=0.156 track_nmae=yaw:0.1884,roll:0.1985,pitch:0.1649 max=0.1985 peak_force=2267.4
reference_lift root_pitch=47.5 leg_pose=neutral drive=position_source_rate support_ratio=0.156 track_nmae=yaw:0.1674,roll:0.1788,pitch:0.1457 max=0.1788 peak_force=2514.4
reference_lift root_pitch=47.5 leg_pose=neutral drive=source_force_rate support_ratio=0.184 track_nmae=yaw:0.1470,roll:0.1755,pitch:0.1464 max=0.1755 peak_force=1800.0
reference_lift root_pitch=47.5 leg_pose=source_retracted drive=position_physics_rate support_ratio=0.157 track_nmae=yaw:0.1884,roll:0.1985,pitch:0.1649 max=0.1985 peak_force=2267.4
reference_lift root_pitch=47.5 leg_pose=source_retracted drive=position_source_rate support_ratio=0.156 track_nmae=yaw:0.1674,roll:0.1788,pitch:0.1457 max=0.1788 peak_force=2514.4
reference_lift root_pitch=47.5 leg_pose=source_retracted drive=source_force_rate support_ratio=0.185 track_nmae=yaw:0.1470,roll:0.1755,pitch:0.1464 max=0.1755 peak_force=1800.0
```

`source_force_rate` はupstream FlyBodyに近づけるため、0.2 msごとに `gain * clip(target-current, -1, 1)` を計算して4 physics step保持する経路です。それでも十分な改善はありません。

脚neutral/retracted差はほぼありません。

### 3. 既に検討・一部除外された仮説

以下を「完全に除外済み」と決めつける必要はありませんが、既存ログでは主因らしさが低下しています。

- 1.09 mgという質量だけが原因
- 脚をretractしていないことだけが原因
- wing actuatorの±30 force clampだけが原因
- root pitch 47.5°の二重適用だけが原因
- FlyGym POSITION shortcutだけが原因

## 重点的に読んでほしいvirtual-flyファイル

- `scripts/embodiment/flybody_runtime.py`
- `scripts/embodiment/flybody_flight_physics.py`
- `scripts/embodiment/flybody_biophysics.py`
- `scripts/embodiment/flybody_measured_wingbeat.py`
- `scripts/embodiment/flybody_reference_lift_calibration.py`
- `scripts/embodiment/flybody_vertical_flight_capability.py`
- `scripts/embodiment/flybody_vertical_flight_diagnosis.py`
- `scripts/embodiment/flybody_muscle_adapter.py`

## 重点比較項目

少なくとも以下をsource FlyBodyとcompiled virtual-flyで比較してください。

1. wing body frame / joint axis / joint ordering
2. wing joint range / springref / armature / stiffness / damping
3. wing inertial geomのsize / pos / quat / mass
4. FlyGym変換後にmembrane meshへ移されたmassをvirtual-flyが正しく除去・復元できているか
5. compiled `body_mass` / `body_inertia` がsourceと単位変換上同値か
6. fluid ellipsoidのsize / pos / quat / `fluidcoef`
7. global air density / viscosity / gravity / timestep
8. cm系FlyBody→mm系FlyGymでのtorque, inertia, density, viscosityの変換則
9. measured wingbeatの軸順・符号・左右翼への適用
10. source WPGのresampling / initialization / qvel初期化 / control cadence
11. reference lift診断そのものが正しい量を測っているか。特にroot accelerationとwhole-body CoM accelerationの混同がないか
12. MuJoCo version / dm_control旧実装とMjSpec/FlyGym 2.1のfluid model semantics差

## 監査上の制約

- 「とりあえずgainを増やす」「fluidcoefを適当に増やす」などの校正で症状だけ消さないでください。
- source FlyBodyとの非同値点を特定することが優先です。
- 根拠のない生物学的パラメータを新設しないでください。
- CNSや学習則は今回の主対象ではありません。reference wingbeatだけでも支持力が不足しているためです。
- 既存の診断結論を追認する必要はありません。診断コード自体の誤りも積極的に疑ってください。

## 出力形式

1. **最有力原因** — 根拠となる具体的コード箇所とsourceとの差
2. **次点の原因候補** — 優先順位付き
3. **診断コードの問題点** — あれば具体的に
4. **原因を一発で分離できる最小実験** — 実装可能な形で
5. **修正案** — sourceとの同値性を回復するもののみ
6. **確信度** — 各主張について高/中/低

最終的には「なぜ現在のvirtual-flyでは公式measured wingbeatでも平均支持力が体重の約0.1〜0.2倍しか出ないのか」を説明してください。
