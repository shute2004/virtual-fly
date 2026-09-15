# 学習プロセスとライブ可視化の分離

## 1. 方針

Flyppy学習プロセスは描画を所有しない。

通常の学習ではFlyBody/MuJoCoとMaleCNS runtimeを学習側で保持し、可視化は別プロセスのobserverとして任意に接続・切断する。observerが存在しない間、viewer用snapshot取得・追加neural read・telemetry JSON更新は行わない。

```text
Flyppy / FlyBody / MaleCNS training
            ^
            | local Unix stream while observer is open
            |
      detached viewer process
            |
            | on-demand telemetry while stream is connected
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

viewer serverはtraining側が所有するローカルUnix stream socketへ接続し、そのstream自体をviewer leaseとして扱う。接続中だけ `publisher.requested()` がtrueになり、viewer用neural read・body snapshot・telemetry publishが有効になる。viewer終了または異常終了でstreamが切断されれば自動的にfalseへ戻る。

control socketのidentityはexperiment path文字列そのものではなく、存在するexperiment directoryのfilesystem identityから導出する。同じ物理directoryを `Desktop` / `desktop` のような異なるcase spellingやsymlink経由で参照しても、同じviewer endpointへ到達する。viewer側は接続失敗時にidentityを再計算するため、viewer先行起動後にexperiment directoryが作成・再作成された場合にも追従できる。

viewer不在時は、学習側はviewer用JSONを書き続けず、body snapshot IPCも追加のviewer-neuron readも実行しない。

`--telemetry` / `VF_POPULATION_TELEMETRY=1` は常時telemetryを強制する診断用overrideであり、通常のviewer利用には不要。

## 4. 身体viewer

`scripts/embodiment/live_body_viewer.py` は独立したFlyBody/MuJoCoモデルを持つ。

training側からviewer要求中だけ公開される `qpos` / `qvel` を読み、自分の `MjData` へコピーして `mj_forward` するだけであり、training physicsをstepしない。

新しいrunで `body.json` が消えた場合は前runのposeを保持せず、`fly.png` を削除して待機状態へ戻る。これにより古いposeの再描画をlive frameと誤認しない。

macOSではMuJoCo viewerの都合でviewer processだけを `mjpython` から起動する。training process自体は通常のPythonでheadless実行する。

## 5. MaleCNS 3D viewer

`visualization/live-neural-viewer.html` はThree.jsで表示する。

静的表示グラフは `scripts/data/prepare_neural_viewer_graph.py` がreleased MaleCNS connectomeから生成する。全約2,560万edgeをブラウザへ送らず、embodiment boundary neuronと強いreleased connectionを含むbounded subgraphを使う。既定budgetは最大2,500 node / 6,000 edgeである。

viewerはtrainingから受け取った `depolarizing_body_ids` を使って発火を表示する。表示上は瞬間的なon/offだけでなく短い残光を与え、telemetry sample間でも活動の時間方向を追いやすくする。

connectionの基礎明度はviewer graphに含まれるreleased `synapse_count` を対数正規化して表現する。presynaptic neuronが発火したconnectionではedgeが一時的に明るくなり、pre→post方向へviewer上の伝播パルスを流す。これは可視化表現であり、個々のsynaptic currentを追加計測しているわけではない。

PAM reward / PPL aversive eventではdopamine neuron群の残光をevent色で強調する。

現在の3D node位置は、side / superclass / nerve等から分類したbrain・optic lobe・VNC領域と決定論的jitterによる模式配置である。MaleCNSの実際の3D morphologyまたはsynapse座標を表すものではない。公式の形態座標を取り込んだ場合のみ解剖学配置へ置換する。

## 6. viewer health

`scripts/embodiment/live_viewer_server.py` は `/api/viewer-health` を提供する。

ここでは少なくとも次を確認できる。

- viewerとtrainerのtelemetry stream接続状態
- 期待するcontrol socket path
- `status.json`
- `body.json`
- `neural.json`
- `fly.png`
- `camera.json`

body rendererが終了した場合、`scripts/dev/view_flyppy.sh` はHTTP serverだけを残して成功状態にせず、body renderer logを表示してviewer全体を失敗扱いにする。

## 7. 起動

学習は通常どおり起動する。viewer用フラグは不要。

```bash
bash scripts/dev/train_flyppy_v3_population.sh
```

学習開始後、見たくなった時だけ別ターミナルでobserverを起動する。

```bash
bash scripts/dev/view_flyppy_v3.sh
```

observerを途中で閉じてもtrainingは継続する。後で再度observerを起動すれば、その時点の学習状態から再び観測できる。
