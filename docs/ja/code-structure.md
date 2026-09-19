# コード構成

[English](../en/code-structure.md) · [日本語](code-structure.md) · [简体中文](../zh-CN/code-structure.md)

この文書では、現在の実装における開発者向けの責務分担を説明します。過去の成果物がすべて現在のコードから生成された、という意味ではありません。

## 現在の学習実行経路

```text
scripts/dev/train_flyppy_population.sh
  -> scripts/embodiment/train_flyppy_population_packed.py   （薄いCLI層）
  -> virtual_fly.training.population_packed                 （実行方式の選択・準備）
  -> virtual_fly.training.population_process                （エピソード全体の進行管理）
       -> virtual_fly.training.flyppy_config                （CLI・設定・来歴の検証）
       -> virtual_fly.training.curriculum*                  （初期条件の調整のみ）
       -> virtual_fly.training.checkpointing                （共有重みのみの保存）
       -> virtual_fly.runtime.neural_bridge                 （Rust神経実行系との通信）
       -> virtual_fly.runtime.packed_*                      （MuJoCoプロセス境界）
            -> virtual_fly.embodiment.factory               （身体・感覚系の構築）
                 -> virtual_fly.embodiment.body             （版管理されたFlyBody接続）
                 -> virtual_fly.embodiment.retina/haltere   （感覚変換）
                 -> virtual_fly.embodiment.periphery        （個別運動ニューロンの末梢状態）
```

`train_flyppy_v3_population.sh`というファイル名は互換性のためだけに残しています。現在の汎用入口は`train_flyppy_population.sh`です。

## 科学的な互換性契約

`virtual_fly.semantics`では、互換性に関わる次のような仕様を識別子として管理します。

- MaleCNS / class-DANの定義
- 神経活動・可塑性の仕様版
- 複数個体学習におけるチェックポイントと神経ステップ数の意味
- 平衡器由来成果物の種類
- 直接レイ視覚で使うK値
- 身体版と、その直線的ではない継承関係

`virtual_fly.reproducibility`は、これらの条件を検証し記録します。別の学習則を定義するモジュールではありません。

## 身体版の系統

身体版の番号は単純な直線継承ではなく、歴史的なラベルです。

```text
v3
└─ v4
   ├─ v5
   ├─ v6
   └─ v7

v8 = v6 + v7のneutral trim
```

実装本体は`virtual_fly.embodiment.body`にあります。`scripts/embodiment/flybody_v*_adapter.py`以下の互換用ファイルは、パッケージ実装を再公開しているだけです。

## 実行時処理と純粋な設定処理

- `virtual_fly.embodiment`: 身体モデル、環境、感覚器、末梢状態
- `virtual_fly.training`: 経験条件調整、共有重み学習、保存・再開
- `virtual_fly.runtime`: 子プロセス、GPU、観察用情報など副作用を伴う境界
- `virtual_fly.playback`: 固定条件での再生・評価
- `virtual_fly.reporting`: 機械可読なレポートと履歴
- `virtual_fly.reproducibility`: ハッシュ、来歴、仕様検証

可能な範囲で、単純なスケジューリングや設定処理はMuJoCo・GPU・子プロセスの寿命管理から切り離し、シミュレーターを起動せず単体検証できるようにしています。

## 過去実装と診断コード

`virtual_fly.legacy`には、現在の実行経路では使わない過去の仕組みを置きます。現在の明確な例はDNg02集団復号器です。

過去実験の再現や原因調査に使うため、`scripts/analysis`や`scripts/embodiment`には診断用スクリプトも残しています。古いスクリプト名を読み込むものもありますが、それらは互換層であり、新しい実装から使うべき入口ではありません。

主な例:

- `train_flyppy_curriculum.py`: 過去の直列学習器
- `train_flyppy_population.py`: 同一プロセス内で動く参照・診断用の複数個体学習器
- `evaluate_flyppy.py`: 過去系統・診断用の評価器
- `scripts/analysis/*`: 調査、監査、性能計測、レポート生成。行動制御器ではない

DLM / DVMの集約表現など、現在の身体接続に残っている較正済み・手書きの末梢力学は、工学的近似であるという理由だけで過去実装扱いにはしません。現在の物理仕様の一部です。

## PythonとRustの境界

Pythonは、実験の進行管理、身体との接続、感覚変換、経験条件調整、来歴記録を担当します。Rustは、MaleCNS全体の神経活動、神経修飾、適格度、重み更新を含む大規模神経実行系を担当します。

両者はbody IDを指定した電流・イベントと、明示的なチェックポイント・更新手順を介して通信します。可視化側は読み取り専用です。

## 身体関連の派生成果物

現在の実行経路で生成する身体関連成果物が、過去実験で使った成果物を上書きしてはいけません。

特に中立トリムは次のように分離しています。

- 過去のv7用: `artifacts/embodiment/wing-pattern-neutral-trim-v1.*`
- 現在の来歴検証済み版: `artifacts/derived/wing-pattern-neutral-trim-v1.*`

`train_flyppy_population.sh`は、現在用の2ファイルがどちらも存在しない場合だけ生成し、型付きの身体設定を通してそのパスを明示的に渡します。過去実験を再生する場合は、明示的な過去成果物利用設定のもとで古い成果物を指定できます。

v7のトリム数値そのものを変更したわけではありません。保存場所を分ける目的は、身体力学を変えることではなく来歴を保護することです。
