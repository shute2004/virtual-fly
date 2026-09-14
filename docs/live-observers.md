# 学習プロセスとライブ可視化の分離

## 1. 方針

Flyppy学習プロセスは描画を所有しない。

通常の学習は、1つのPythonプロセス・1つのFlyBody/MuJoCoインスタンス・1つのMaleCNS runtimeを複数episodeにわたって保持する。可視化は別プロセスのobserverとして任意に接続・切断する。

```text
Flyppy / FlyBody / MaleCNS training
            |
            | atomic telemetry files
            v
artifacts/experiments/flyppy-v1/live/
            |
            +--> detached MuJoCo body viewer
            |
            +--> browser MaleCNS activity viewer
```

viewerの起動・終了は学習状態、物理状態、神経状態、可塑性状態を変更してはならない。

## 2. 学習runtime

`scripts/dev/train_flyppy.sh` は `scripts/embodiment/train_flyppy_curriculum.py` を起動する。

このrunnerではepisodeごとにRust/GPU processを作り直さない。学習済みsynaptic weightsはGPU runtime内に保持し、episode境界では短期神経状態だけをresetする。

full CNS checkpointは毎step・毎episodeではなく、既定では4 episodeごととrun終了時に保存する。

GPU backend選択時は起動ログへ実際のadapter名を含む `neural_backend=gpu:...` を出力する。CPU backendはRust/Rayonのdata parallel runtimeを使用する。

## 3. telemetry

`scripts/embodiment/live_telemetry.py` は小さなJSONファイルを一時ファイルからatomic replaceして公開する。

- `status.json`: backend、episode、curriculum状態
- `body.json`: MuJoCo `qpos` / `qvel`、gate/event、末梢運動状態
- `neural.json`: 可視化対象MaleCNS body IDの活動、DAN event

telemetryは制御入力ではない。observerからtrainingへ戻る通信路は持たない。

## 4. 身体viewer

`scripts/embodiment/live_body_viewer.py` は独立したFlyBody/MuJoCoモデルを持つ。

training側の `qpos` / `qvel` を読み、自分の `MjData` へコピーして `mj_forward` するだけであり、training physicsをstepしない。

macOSではMuJoCo viewerの都合でviewer processだけを `mjpython` から起動する。training process自体は通常のPythonでheadless実行する。

## 5. MaleCNS 3D viewer

`visualization/live-neural-viewer.html` はThree.jsで表示する。

静的表示グラフは `scripts/data/prepare_neural_viewer_graph.py` がreleased MaleCNS connectomeから生成する。全約2,560万edgeをブラウザへ送らず、embodiment boundary neuronと強いreleased connectionを含むbounded subgraphを使う。

発火した表示対象neuronを明るくし、そのneuronをpresynaptic endpointに持つ表示対象connectionを同じframeで発光させる。

現在の3D node位置は、side / superclassと決定論的jitterを用いた**模式配置**である。MaleCNSの実際の3D morphologyまたはsynapse座標を表すものではない。将来、公式の形態座標を取り込んだ場合のみ解剖学配置へ置換する。

## 6. 起動

学習:

```bash
bash scripts/dev/train_flyppy.sh --episodes 12 --trajectory-stride 10
```

別ターミナルからobserver:

```bash
bash scripts/dev/view_flyppy.sh
```

observerを途中で閉じてもtrainingは継続する。training途中でobserverを起動してよい。
