# 基礎資料・外部資産

この文書は、実装で参照する一次資料、公式データ、主要ソフトウェアを整理するための入口である。

実際に利用する際はURLだけでなく、論文、データセットの版、ライセンス、取得日時を来歴記録へ残す。

## 1. MaleCNS

### Male CNS Connectome project

- プロジェクト: https://male-cns.janelia.org/
- ダウンロード / プログラムからの取得: https://male-cns.janelia.org/download/
- データセット: `male-cns:v1.0`
- neuPrint: https://neuprint.janelia.org/
- 論文付属リポジトリ: https://github.com/flyconnectome/2025malecns

公式ダウンロードページでは、`neuprint-python`によるAPIアクセスと一括データの両方が提供されている。データセットのライセンスはCC-BYである。

### 論文

Berg et al. (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature.

- https://www.nature.com/articles/s41586-026-10735-w

全脳と腹神経索を連続したCNSとして扱うための基礎資料。

### MaleCNS視葉の網膜対応

公式注釈の`assignedOlHex1` / `assignedOlHex2`を、視葉カラムの網膜対応座標として使う。視覚入力では、`body ID`の並び順や配列上の添字を空間位置として扱わない。

MaleCNS公式メディアには、R1-R6から下降性ニューロンまでの視覚―運動経路の例も公開されている。

- https://male-cns.janelia.org/media/

## 2. ショウジョウバエ初期視覚系

### 完全視覚系コネクトーム

Nern et al. (2025), *Connectome-driven neural inventory of a complete visual system*, Nature.

- https://www.nature.com/articles/s41586-025-08746-0

この資料では、laminaが撮像体積に完全には含まれていないため、再構築されたLaiおよびR1-R6の数が生物学的な総数を過小評価すると明記されている。したがって`virtual-fly`では、欠損したR1-R6を幾何学的に作り足したり、各カラムが必ず6細胞になるよう補完したりしない。公開MaleCNSに実際に存在するR1-R6だけを`observed`として使う。

### 神経重複

Langen et al. (2015), *The Developmental Rules of Neural Superposition in Drosophila*.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4646663/

R1-R6では、同じ個眼内の6細胞を1点として平均するのではなく、異なる近傍個眼に由来し同じ視軸を見るR1-R6が、同じラミナカートリッジへ収束する。網膜対応入力を実装するときの重要な配線原理である。

Juusola et al. による電気生理学的手法の概要:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4993232/

ラミナカートリッジが網膜上の小さな視野領域を処理し、R1-R6がL1-L3などへヒスタミン作動性出力を送ることの参照資料。

### 方向選択性とカラム配列

Fisher et al. (2015/2016), *Complementary mechanisms create direction selectivity in the fly*.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4978522/

T4/T5の方向選択性は、視葉のカラム配列上にある局所回路で形成される。`virtual-fly`では、外部コードがT4/T5相当の運動特徴を先に計算して注入するのではなく、可能な限りR1-R6から実回路を通して生じさせる。

## 3. FlyBody

- リポジトリ: https://github.com/TuragaLab/flybody
- MuJoCo Menagerie内の複製: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody

ショウジョウバエの解剖学的な3D身体とMuJoCo物理を利用する主要候補。

重要: 本プロジェクトでは、FlyBodyに付属する強化学習方策を学習機構として使用しない。利用するのは身体形状、物理、アクチュエータなどであり、運動指令は`virtual-fly`のCNSから与える。

## 4. FlyGym / NeuroMechFly

- 文書: https://neuromechfly.org/
- 導入方法: https://neuromechfly.org/installation/
- チュートリアル: https://neuromechfly.org/tutorials/

感覚、身体、MuJoCo統合、FlyBody利用、GPU実行などの参考実装として使う。

FlyGym 2.xは旧APIと互換ではないため、依存する版を明示して固定する。

FlyGym標準の複眼`Retina`は合成六角格子を使うため、MaleCNS個体のbody IDと網膜位置の対応そのものとはみなさない。現在のMaleCNS境界では、FlyBodyの生の眼カメラを局所光源として使い、MaleCNS側の実カラム座標を優先する。

## 5. MuJoCo

- プロジェクト: https://mujoco.org/
- リポジトリ: https://github.com/google-deepmind/mujoco
- Menagerie: https://github.com/google-deepmind/mujoco_menagerie

身体物理に使用する主要候補。

## 6. MaleCNSへのアクセス

公式Pythonアクセス例は概ね次の形になる。

```python
from neuprint import Client

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token="...",
)
```

実装では、認証トークンをコードや設定ファイルへ登録しない。

## 7. 今後整理する一次文献

実装を追加する前に、少なくとも次の項目についてショウジョウバエの一次研究を整理する。

- 膜電位・発火特性
- 化学シナプスの伝達特性
- 電気シナプス
- 神経伝達物質と受容体
- キノコ体可塑性
- PAM / PPL1などのドーパミン作動性回路
- 報酬性・嫌悪性条件づけ
- STDPなどの活動依存可塑性
- 恒常性可塑性
- 構造可塑性
- 自発運動
- 飛翔の中枢パターンと下降性制御
- 翼運動ニューロンと飛翔筋
- 複眼の光受容変換
- 固有感覚・機械感覚

各実装パラメータについて、「一般的な神経科学モデルだから」という理由だけで値を流用せず、ショウジョウバエで利用できるデータを優先する。

## 8. 参照時のルール

1. ブログや解説記事より、一次論文と公式データを優先する。
2. 別種の動物由来の値を使う場合は、`assumed`または`inferred`として明示する。
3. データセット更新へ自動追従せず、実験ごとに版を固定する。
4. 実測コネクトームと、そこから生成した仮想個体の状態を別物として保存する。
5. 外部ソフトウェアに含まれる強化学習やニューラルネットワークの制御器を、明示せずCNSの代わりに使わない。
6. 実際の局所回路が担う特徴抽出を、CNS外の便利な特徴量計算で置き換えない。
