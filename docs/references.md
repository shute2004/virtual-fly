# 基礎資料・外部資産

この文書は、実装で参照する一次資料・公式データ・主要ソフトウェアを固定するための入口である。

URLだけでなく、実際に利用するときは論文・データセットのバージョン・ライセンス・取得日時をmanifestへ記録する。

## 1. MaleCNS

### Male CNS Connectome project

- Project: https://male-cns.janelia.org/
- Download / programmatic access: https://male-cns.janelia.org/download/
- Dataset: `male-cns:v1.0`
- neuPrint: https://neuprint.janelia.org/
- Publication supplemental repository: https://github.com/flyconnectome/2025malecns

公式ダウンロードページでは、`neuprint-python` によるAPIアクセスと一括データが提供されている。データセットはCC-BY。

### Publication

Berg et al. (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature.

- https://www.nature.com/articles/s41586-026-10735-w

全脳と腹神経索を連続したCNSとして扱うための基礎資料。

### MaleCNS optic-lobe retinotopy

公式annotationの `assignedOlHex1` / `assignedOlHex2` をoptic-lobe columnのretinotopic座標として使用する。視覚入力実装ではbody IDの任意順序や配列indexを空間位置として扱わない。

MaleCNS公式mediaにはR1-R6からdescending neuronまでのvisual-motor pathway例も公開されている。

- https://male-cns.janelia.org/media/

## 2. ショウジョウバエ初期視覚系

### Neural superposition

Langen et al. (2015), *The Developmental Rules of Neural Superposition in Drosophila*.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4646663/

R1-R6では、同じommatidium内の6細胞を1点として平均するのではなく、異なる近傍ommatidiaに由来し同じvisual axisを見るR1-R6が同じlamina cartridgeへ収束する。retinotopic入力を実装するときの重要な配線原理。

Juusola et al. / electrophysiology protocol overview:

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4993232/

lamina cartridgeがretinotopicに小さな視野領域を処理し、R1-R6がhistaminergic outputをL1-L3等へ送ることの参照。

### Direction selectivity / columnar raster

Fisher et al. (2015/2016), *Complementary mechanisms create direction selectivity in the fly*.

- https://pmc.ncbi.nlm.nih.gov/articles/PMC4978522/

T4/T5方向選択性はoptic-lobeのcolumnar raster上の局所回路で形成される。virtual-flyでは、外部コードがT4/T5相当の運動特徴を先に計算して注入するのではなく、可能な限りR1-R6から実回路を通して生じさせる。

## 3. FlyBody

- Repository: https://github.com/TuragaLab/flybody
- MuJoCo Menagerie copy: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody

ショウジョウバエの解剖学的3D身体とMuJoCo物理を利用する第一候補。

重要: 本プロジェクトではFlyBodyに付属するRL方策を学習機構として使用しない。身体・物理・アクチュエータ等を利用し、運動指令はvirtual-flyのCNSから与える。

## 4. FlyGym / NeuroMechFly

- Documentation: https://neuromechfly.org/
- Installation: https://neuromechfly.org/installation/
- Tutorials: https://neuromechfly.org/tutorials/

感覚・身体・MuJoCo統合、FlyBody利用、GPUシミュレーション等の参考実装として利用する。

FlyGym 2.xは旧APIと互換ではないため、依存バージョンを明示して固定する。

FlyGymの標準compound-eye `Retina` は合成hex gridを用いるため、MaleCNS個体のretinotopic body-ID対応そのものとしては扱わない。現在のMaleCNS境界ではFlyBodyのraw eye cameraを局所光源として使用し、MaleCNS側の実column座標を優先する。

## 5. MuJoCo

- Project: https://mujoco.org/
- Repository: https://github.com/google-deepmind/mujoco
- Menagerie: https://github.com/google-deepmind/mujoco_menagerie

身体物理の第一候補。

## 6. MaleCNSアクセス

公式Pythonアクセス例では概ね次の構成になる。

```python
from neuprint import Client

client = Client(
    "https://neuprint.janelia.org",
    dataset="male-cns:v1.0",
    token="...",
)
```

実装ではトークンをコード・設定ファイルへコミットしない。

## 7. 今後追加する一次文献

実装開始前に、少なくとも以下についてショウジョウバエの一次研究を整理する。

- 膜電位・発火特性
- 化学シナプスの伝達特性
- 電気シナプス
- 神経伝達物質と受容体
- キノコ体可塑性
- PAM / PPL1等のドーパミン作動性回路
- appetitive / aversive conditioning
- STDP等の活動依存可塑性
- homeostatic plasticity
- structural plasticity
- spontaneous locomotion
- flight central pattern / descending control
- wing motor neurons / flight muscles
- compound-eye transduction
- proprioception / mechanosensation

各実装パラメータについて、「何となく一般的な神経科学モデル」を流用せず、ショウジョウバエで利用可能なデータを優先する。

## 8. 参照時のルール

1. ブログ・解説記事より一次論文・公式データを優先する。
2. 別種の動物由来の値を使う場合は `assumed` または `inferred` として明示する。
3. データセット更新時に自動追従しない。実験単位で版を固定する。
4. 実測コネクトームと、そこから生成した仮想個体の状態を別物として保存する。
5. 外部ソフトウェアに含まれるRL・NN制御器を、明示せずCNSの代わりに使用しない。
6. 実際の局所回路が担うfeature extractionを、CNS外の便利な特徴量計算で代替しない。
