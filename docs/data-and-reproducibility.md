# データ来歴と再現性

## 1. 基準データ

初期基準は HHMI Janelia / collaborators による成体オスショウジョウバエ全CNSコネクトーム `male-cns:v1.0` とする。

公式配布では neuPrint API と一括ダウンロードが提供されている。データセット本体は CC-BY とされている。

実装では、取得時点で必ず以下を記録する。

- dataset name
- dataset version
- source URL
- acquisition date
- upstream license
- upstream citation
- file hashes
- conversion tool version
- conversion config

## 2. 生データを直接編集しない

外部データは `source snapshot` として読み取り専用に扱う。

```text
upstream source
      ↓
verified local snapshot
      ↓
normalized internal dataset
      ↓
virtual individual state
```

学習・可塑性による変更は、元コネクトームへ書き戻さない。

## 3. データ層

### Layer A: upstream

取得した原データまたはAPI応答。

### Layer B: normalized

内部スキーマへ正規化した静的データ。

例:

- neurons
- synapses / aggregated connections
- annotations
- morphology references
- provenance

### Layer C: individual state

仮想個体ごとの変更可能状態。

- neural state
- synaptic functional state
- structural deltas
- neuromodulation state
- body state
- experiment history

## 4. 来歴ラベル

値・パラメータ・接続には可能な範囲で次の来歴を付ける。

| label | 意味 |
|---|---|
| `observed` | 実測または上流データに直接存在 |
| `literature` | 文献から採用 |
| `inferred` | 実測値から推定 |
| `assumed` | 未知部分を埋めるための仮定 |
| `calibrated` | 検証実験に合わせて較正 |

論文・README・可視化では、これらを混同しない。

## 5. 実験manifest

各実験runには機械可読manifestを生成する。

例:

```yaml
experiment_id: exp-...
fly_id: fly-...
parent_checkpoint: ...
code_commit: ...
connectome:
  dataset: male-cns:v1.0
  normalized_snapshot_hash: ...
neural_model:
  name: ...
  version: ...
plasticity_model:
  name: ...
  version: ...
body:
  model: flybody
  version: ...
environment:
  name: flyppy
  config_hash: ...
rng:
  seed: ...
```

## 6. 再現性レベル

### R0: 設定再現

同じ入力データ・設定・コード版を特定できる。

### R1: 決定論再現

同一backend・同一環境では同一結果を再生できる。

### R2: backend間整合

CPU / GPU / WebGPU等で、許容誤差内の同等結果を得られる。

### R3: 統計再現

非決定的な要素がある場合でも、複数runで同等の統計的結論を得られる。

すべての実験がR2を満たす必要はないが、どのレベルを満たすか明記する。

## 7. 外部資産

外部資産は原則としてリポジトリへ直接vendorしない。

候補:

- MaleCNS
- FlyBody
- FlyGym / NeuroMechFly
- MuJoCo

必要な場合は取得スクリプトと固定バージョンを用意する。

## 8. 大容量生成物

次はGit管理しない。

- raw connectome dumps
- EM volumes
- large meshes
- checkpoints
- trajectory datasets
- rendered videos
- profiling traces

代わりにmanifestとハッシュをGitで管理する。

## 9. 公開ブラウザ実験

ブラウザから返るデータは信用境界の外にあるものとして扱う。

各結果には最低限以下を含める。

- experiment package version
- initial state hash
- config hash
- execution backend
- result hash
- summary metrics

重要な結果はサーバ側または複数独立クライアントで再実行する。

## 10. 個人データ

公開実験時、神経科学実験に不要な個人情報・ブラウザ識別情報は収集しない。

計算参加に必要な実行環境情報を取得する場合も、目的と保持範囲を明示する。
