# 翼運動ニューロンと末梢筋の境界

[English](../en/wing-motor-boundary.md) · [日本語](wing-motor-boundary.md) · [简体中文](../zh-CN/wing-motor-boundary.md)

## 1. 目的

Flyppy学習でCNSの外側に行動決定器を置かないため、MaleCNSから身体への境界を、公開されている個々の翼運動ニューロンまで下げる。

使用しない経路:

```text
DNg02集団活動
  -> 平均発火率
  -> 左右差・全体平均
  -> 翼振幅
```

現在の学習経路:

```text
FlyBodyの眼
  -> R1-R6 body IDへの局所電流
  -> MaleCNS全体
  -> VNCの運動前回路
  -> 個々の翼運動ニューロンの発火
  -> 個々の運動単位・神経筋状態
  -> 同定済み翼筋
  -> 仮想筋によるトルク
  -> FlyBody / MuJoCo
```

これは、運動ニューロン群の活動ベクトルから行動を選ぶ復号処理ではない。神経から筋へ、筋から関節力学へつながる末梢過程である。

## 2. MaleCNSの翼運動ニューロン一覧と使用対象

`scripts/data/prepare_wing_motor_map.py`は、MaleCNS v1.0で次に該当する公開ニューロンをbody ID単位で抽出する。

```text
superclass = vnc_motor
subclass   = wm
```

2026年9月14日に取得した実データでは次の内訳だった。

- 67ニューロン
- 左33 / 右34
- 注釈済み型26種類
- `identified` 38
- `identified_group` 22
- `identified_variable` 2
- `putative` 2
- `unknown` 3

初期の学習境界では、`identified`と`identified_group`の60ニューロンだけを使う。残る7ニューロンは、対応先を推測して接続しない。

代表的な除外例:

- `MNwm35 -> iii4`: `putative`
- `MNwm36`: 対応する筋が未解決
- `tpn`: tp1 / tp2への神経支配が可変
- 型が同定されていない公開翼運動ニューロン

DLM / DVMのように複数細胞をまとめた型でも、追加の証拠なしにbody IDの順番だけで各筋線維へ割り当てない。

## 3. 実装済みの末梢層

### 3.1 個別運動ニューロンの発火から運動単位状態へ

`scripts/embodiment/wing_muscle_periphery.py`が実装する。

- 神経ブリッジから60個のbody IDの発火を個別に取得する
- `body ID`ごとに独立した活性状態を保持する
- 主動力筋、直接操舵筋、間接制御筋で異なる時定数を使う
- 要求したbody IDが1つでも返ってこなければ、そのまま続行せず失敗として停止する
- 集団平均発火率や左右平均を作らない

複数の運動単位が同じ筋群へ入る場合は、各運動単位を独立に更新した後、筋における物理的な動員として飽和付きで合成する。これは行動を選ぶ復号器ではなく、末梢での力発生が一つの筋へ収束する過程である。

現在の時定数と1発火あたりの動員量は`calibrated`な初期較正パラメータであり、MaleCNSで直接実測された定数として扱わない。

### 3.2 非同期主動力筋

DLM / DVMは非同期型の間接飛翔筋であり、運動ニューロンが約200 Hzの各羽ばたきを1回ずつ直接指令するモデルにはしない。

`scripts/embodiment/flybody_muscle_adapter.py`では、DLM / DVMの低周波な活性を、胸部飛翔振動系が利用できる主動力へ変換する。羽ばたき位相は、伸張活性と胸部共振を近似する末梢の機械状態であり、CNSが選ぶ行動ではない。

エピソード開始時点ですでに飛行中である条件を表すため、主動力筋だけには初期活性を与える。これは前進初速度や初期羽ばたき位相と同じく、エピソード開始条件であり、各時間刻みに外部から与える運動指令ではない。

### 3.3 操舵筋からトルクへ

FlyBodyには解剖学的な翼筋そのものが存在しないため、現時点では各筋をそのままMJCFの筋要素へ接続できない。そのため第一段階では、`qfrc_applied`を使う仮想筋層を置く。

現在、トルクへ作用させる直接操舵筋は、定性的な作用方向を比較的強く制約できるものだけに限る。

- b1: ストローク振幅を増やす側
- b2: ストローク振幅を増やす側
- b3: basalar pairに対する拮抗側
- i1: ストローク振幅を減らす側

数値利得は、実測されたモーメントアームではなく`calibrated`として扱う。

その他の同定済み操舵筋・間接制御筋も活性状態までは保持するが、作用方向を推測してトルクを発生させることはしない。

### 3.4 FlyBodyへの接続

旧DNg02変換器は過去の物理診断用として残すが、Flyppyの現在の学習経路からは外している。

`FlyBodyMuscleAdapter`は、継承元にある6個の理想化された位置制御アクチュエータを、各物理時間刻みで現在角へ合わせて中立化する。これにより、旧来の解析的な位置指令だけで行動が発生する経路を止める。そのうえで、仮想筋のトルクを翼の自由度へ加える。

## 4. 報酬・嫌悪刺激

報酬・嫌悪刺激と、運動出力の境界は別の仕組みである。

外部コードが神経実行系へ直接渡してよいのは、実在するDANの`body ID`への電流刺激だけである。

現在のFlyppy実験では次のように扱う。

```text
ゲート通過
  -> 公開PAM01（PAM-gamma5）DANへ電流刺激

衝突
  -> 公開PPL1嫌悪関連集合（PPL101/PPL103/PPL106）DANへ電流刺激
```

`reward=+1`、`punishment=-1`のような数値、単一の数値報酬、重みの直接更新は存在しない。刺激後のドーパミン状態と可塑性は、MaleCNSの接続上を時間発展する神経実行系の中で生じる。

PAM01の内部にも生理学的な多様性があり得るため、「PAM01の全細胞が自然条件でも完全に同一の報酬信号を表す」とは主張しない。嫌悪側も同様で、PPL1集合は、公開MaleCNS上に実在するDANを使った実験条件として扱う。

PPL101単独では、実際の衝突軌跡の一部で、イベント直前の適格度と投射先が重ならず、追加の可塑変化が0になることを確認した。一方、PPL101 / PPL103 / PPL106の6細胞集合では、同じ軌跡に対して4,935本の可塑性対象辺にイベント固有の差が生じた。そのため、単一コンパートメントへの依存を避ける最小変更として、この6細胞集合へ拡張した。外部の数値罰や目標重みは追加していない。

## 5. 学習開始前の確認

`scripts/dev/train_flyppy.sh`は、起動前に次を再生成・検査する。

1. 報酬・嫌悪刺激用および診断用の神経群
2. 個別翼運動ニューロン対応表
3. R1-R6の網膜対応表
4. 翼の神経筋境界の簡易動作確認
5. Flyppy閉ループ学習

DNg02集団平均に依存する保護処理は、学習経路を個別運動ニューロン境界へ移したため削除した。

## 6. 残る限界

現在の境界は、「実MaleCNS → 個別運動ニューロン → 解剖学的に同定された筋」まではデータや文献に基づく。一方、「筋 → FlyBodyの翼関節」は仮想筋による近似である。

今後、より高い生体忠実度を得るには、筋ごとの付着点、モーメントアーム、活性―力特性を導入し、FlyBody内に解剖学的な筋モデルを追加する必要がある。ただし、不明な値を行動成績が良くなるよう逆算して埋めてはいけない。

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

## 8. 来歴の区分

- `observed`: MaleCNSのbody ID、型、左右、公開CNS接続。
- `literature`: 翼運動ニューロンと筋標的、主動力筋・直接操舵筋・間接制御筋の分類、DLM / DVMの非同期飛翔筋特性、b1 / b2 / b3 / i1の定性的作用。
- `putative`: MNwm35 → iii4。
- `unknown`: MNwm36、型未同定の運動ニューロン、まとめて注釈された型の内部での個別筋線維対応。
- `calibrated`: 末梢活性の時定数、1発火あたりの動員量、胸部振動系との結合、仮想筋の利得、トルク上限。
