# 学習プロセスとライブ可視化の分離

## 1. 方針

Flyppy学習プロセスは描画を所有しない。

通常の学習ではFlyBody/MuJoCoとMaleCNS runtimeを学習側で保持し、可視化は別プロセスのobserverとして任意に接続・切断する。observerが存在しない間、viewer用snapshot取得・追加neural read・telemetry JSON更新は行わない。

```text
Flyppy / FlyBody / MaleCNS training
            ^
            | viewer lease heartbeat only while observer is open
            |
      detached viewer process
            |
            | on-demand telemetry while lease is alive
            v
artifacts/experiments/flyppy-v3/live/
            |
            +--> detached MuJoCo body viewer
            |
            +--> browser MaleCNS activity viewer
```

viewerの起動・終了は学習状態、物理状態、神経状態、可塑性状態を変更してはならない。

## 2. 学習runtime

productionのpopulation学習は `scripts/dev/train_flyppy_v3_population.sh` から起動する。

shared MaleCNS runtimeと各slotのFlyBody/MuJoCo状態はviewerとは独立して動作する。viewerが起動していなくても学習は通常どおり進み、viewerを後から起動するために学習を再起動する必要はない。

full CNS checkpointはviewerとは無関係にproduction trainerのcheckpoint cadenceで保存される。

## 3. on-demand telemetry

`scripts/embodiment/live_telemetry.py` はviewerが存在する間だけ、小さなJSONファイルを一時ファイルからatomic replaceして公開する。

- `status.json`: backend、episode、curriculum状態
- `body.json`: MuJoCo `qpos` / `qvel`、gate/event、末梢運動状態
- `neural.json`: 可視化対象MaleCNS body IDの活動、DAN event

viewerはローカルUnix datagram socketへ短いlease heartbeatを送る。この通信は「観測を開始・継続・終了する」という要求だけで、学習内容や物理状態を変更する制御入力ではない。

viewer不在時は、学習側はviewer用JSONを書き続けず、body snapshot IPCも追加のviewer-neuron readも実行しない。viewer終了時には明示的なstop packetで即座に停止し、viewerが異常終了した場合もlease timeoutで自動停止する。

`--telemetry` / `VF_POPULATION_TELEMETRY=1` は常時telemetryを強制する診断用overrideであり、通常のviewer利用には不要。

## 4. 身体viewer

`scripts/embodiment/live_body_viewer.py` は独立したFlyBody/MuJoCoモデルを持つ。

training側からviewer要求中だけ公開される `qpos` / `qvel` を読み、自分の `MjData` へコピーして `mj_forward` するだけであり、training physicsをstepしない。

macOSではMuJoCo viewerの都合でviewer processだけを `mjpython` から起動する。training process自体は通常のPythonでheadless実行する。

## 5. MaleCNS 3D viewer

`visualization/live-neural-viewer.html` はThree.jsで表示する。

静的表示グラフは `scripts/data/prepare_neural_viewer_graph.py` がreleased MaleCNS connectomeから生成する。全約2,560万edgeをブラウザへ送らず、embodiment boundary neuronと強いreleased connectionを含むbounded subgraphを使う。

発火した表示対象neuronを明るくし、そのneuronをpresynaptic endpointに持つ表示対象connectionを同じframeで発光させる。

現在の3D node位置は、side / superclassと決定論的jitterを用いた模式配置である。MaleCNSの実際の3D morphologyまたはsynapse座標を表すものではない。将来、公式の形態座標を取り込んだ場合のみ解剖学配置へ置換する。

## 6. 起動

学習は通常どおり起動する。viewer用フラグは不要。

```bash
bash scripts/dev/train_flyppy_v3_population.sh
```

学習開始後、見たくなった時だけ別ターミナルでobserverを起動する。

```bash
bash scripts/dev/view_flyppy_v3.sh
```

observerを途中で閉じてもtrainingは継続する。後で再度observerを起動すれば、その時点の学習状態から再び観測できる。
