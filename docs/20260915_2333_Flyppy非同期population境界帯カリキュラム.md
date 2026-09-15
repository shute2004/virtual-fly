# Flyppy非同期population境界帯カリキュラム

## 1. 目的

Part4開始時点のproduction trainerは、shared-weight population化後も旧来のadaptive curriculumを使っていた。

この方式では各episode終了時に即座にspawn条件を更新する。

```text
成功 -> 難化
失敗 -> 易化
```

非同期populationではepisode長がslotごとに異なるため、短時間で衝突するslotの結果が先に、かつ多数反映される。長く飛行して成功するslotの結果が返る前に、後続episodeのspawn条件が何度も変わり得る。

Part3では単一個体の第2ゲート訓練でこの1 episode単位のping-pongを問題視し、batch型boundary-bandを導入していた。本変更は、その考え方を現在のshared-weight population runtimeへ戻す。

神経ダイナミクス、PAM/PPL刺激、plasticity則、shared-weight transaction semanticsは変更しない。変更対象はepisode reset時の初期条件選択とcurriculum集計だけである。

## 2. productionの新しい既定

`scripts/dev/train_flyppy_v3_population.sh` の既定を次へ変更する。

```text
curriculum mode: boundary-band
batch size: 24
harden threshold: 0.80
ease threshold: 0.40
```

旧adaptiveは `VF_FLYPPY_CURRICULUM_MODE=adaptive` またはCLI指定で引き続き利用できる。

## 3. adaptive checkpointからの移行

既存のPart3継続個体を捨てない。

boundary-band stateがまだ存在しない場合、現在adaptive conditionを中心に、旧adaptive policyの「次に成功した場合」と「次に失敗した場合」の2条件をそれぞれhard/easy端点として初回bandを作る。

Part4切替直前のproduction stateは概ね:

```text
current: x=5.0490, z=11.1703, vx=425.0
hard:    x=4.7520, z=11.0218, vx=412.5
easy:    x=5.1975, z=11.3188, vx=437.5
target:  x=0.0000, z=8.9100,  vx=300.0
```

したがって、突然別の課題条件へ飛ぶのではなく、現在条件の近傍からbatch学習へ移行する。

## 4. async用attempt identity

単一個体版boundary-bandは `attempts_in_batch` の連番だけで十分だった。

非同期populationでは完了順がlaunch順と一致しないため、各batch内のconditionに明示的な `attempt_index` を付ける。

```text
0 .. batch_size-1
```

curriculum stateは `completed_attempt_indices` を保持する。これにより:

- 完了順が逆転しても同一attemptを二重計上しない
- runを8 episodeで終了し、同じexperimentを16 episode追加するようなpartial-batch resumeでも未完attemptだけを継続する
- batch完了時にのみattempt setを空へ戻す

ことができる。

実測resume smokeでは、8 episode + 16 episodeの2runを跨いでattempt ID 0..23が重複なく1回ずつ消費された。

## 5. condition setは固定、slotへの割当は非同期

batch開始時のhard/easy endpointと24個のease level集合は固定する。

24 episodeの既定分布は従来boundary-bandと同じ:

```text
ease=0.00: 4
ease=0.25: 5
ease=0.50: 6
ease=0.75: 5
ease=1.00: 4
```

ただしattemptを特定slotへ固定はしない。空いたslotへ未消費attemptを渡す。

理由はwall-clock throughputである。attemptを `attempt % slot` で固定したsmokeでは、短いepisodeのslotが担当分を早く使い切って待機し、約19.3 aggregate control steps/sまで低下した。非同期割当では同条件系のsmokeで約23.4 steps/sまで回復した。

shared-weight runtime自体もcommit順・stalenessを持つ非同期学習であるため、ここでは全実行順の決定論化ではなく、**batch中に難易度そのものを動かさないこと**を保証対象とする。

## 6. fast-failing slotによる成功率バイアスへの対処

現在のphysical slotは `seed + slot_id` のFlyppy courseを持つ。

実測するとseedごとのepisode長が大きく異なり、短時間でgate collisionするslotは24 episode中に多数のepisodeを消費する一方、長く飛ぶslotは1回程度しか完了しない場合がある。

したがって単純なraw episode成功率だけでbandを更新すると、fast-failing course seedがcurriculumを支配する。

production boundary-bandでは次を両方記録する。

```text
raw_success_rate
    = 全episodeを等重みした成功率

group_success_rate[slot]
    = 各slot/course seed内の成功率

success_rate used for curriculum
    = group_success_rateをslot間で等重み平均
```

smoke例:

```text
raw episode success: 2/24 = 0.083
slot/course-seed equal-weight success: 0.500
```

rawだけなら`easier`になるが、seed等重みでは`hold`となった。

この変更はtraining experienceそのものを再重み付けしない。shared CNSへ入るPAM/PPLイベントは実際に起きたepisodeのままであり、変更するのは次batchのspawn難易度を決める統計だけである。

## 7. completion-order依存の除去範囲

boundary-bandでは以下を完了順に依存させない。

- hard/easy endpoint
- batch内condition集合
- attemptの二重計上防止
- batch終了判定
- batch成功率の集計
- harden / hold / ease判定

`consecutive_failures` は旧adaptive専用指標なので、boundary-band中は0へ固定する。

一方、shared-weight runtimeのcommit順とversion stalenessは従来どおり非同期である。これは別の学習schedule semanticsであり、本変更では触らない。

## 8. 検証

### unit test

`tests/test_curriculum.py` で以下を確認する。

- 24 episodeのease level比率
- harden / ease
- explicit attempt IDの範囲検証
- duplicate attempt拒否
- outcome完了順を逆転してもband結果が一致
- fast groupがepisode数だけでbatch判定を支配しないequal-group aggregation

### fresh 24-episode smoke

実MaleCNS GPU runtime + packed FlyBody + direct-ray K=13で24 episodeを実行し、以下を確認した。

- trainer完走
- attempt ID 0..23を全て1回消費
- batch完了は24 outcome後の1回のみ
- batch state reset
- shared-weight commit継続

### partial resume smoke

8 episodeで一度checkpointを保存し、同experimentへ16 episode追加した。

- global version 8からresume
- 既完了attemptを再利用しない
- 合計24 attemptが0..23を一度ずつ消費
- 最後にbatchが正常完了

## 9. 今後の評価

この変更だけで学習成功を主張しない。

production continuation checkpointで24 episode単位のtrainingを行い、`flyppy-v3-fixed-v1` 固定評価suiteをbatch前後で比較する。

確認対象は:

- first gate / second gate成功率
- collision率
- altitude loss
- seed別挙動
- 初期MaleCNSとの差

である。

固定評価が改善しない場合は、curriculumより先へ進み、PAM/PPL下のplasticity分布、motor utilization、async shared-weight experience biasを分解する。
