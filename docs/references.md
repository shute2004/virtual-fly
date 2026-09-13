# 基礎資料・外部資産

この文書は、実装で参照する一次資料・公式データ・主要ソフトウェアを固定するための入口である。

URLだけでなく、実際に利用するときは論文・データセットのバージョン・ライセンス・取得日時をmanifestへ記録する。

## 1. MaleCNS

### Male CNS Connectome project

- Project: https://male-cns.janelia.org/
- Download / programmatic access: https://male-cns.janelia.org/download/
- Dataset: `male-cns:v1.0`
- neuPrint: https://neuprint.janelia.org/

公式ダウンロードページでは、`neuprint-python` によるAPIアクセスと一括データが提供されている。データセットはCC-BY。

### Publication

Berg et al. (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature.

- https://www.nature.com/articles/s41586-026-10735-w

全脳と腹神経索を連続したCNSとして扱うための基礎資料。

## 2. FlyBody

- Repository: https://github.com/TuragaLab/flybody
- MuJoCo Menagerie copy: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody

ショウジョウバエの解剖学的3D身体とMuJoCo物理を利用する第一候補。

重要: 本プロジェクトではFlyBodyに付属するRL方策を学習機構として使用しない。身体・物理・アクチュエータ等を利用し、運動指令はvirtual-flyのCNSから与える。

## 3. FlyGym / NeuroMechFly

- Documentation: https://neuromechfly.org/
- Installation: https://neuromechfly.org/installation/
- Tutorials: https://neuromechfly.org/tutorials/

感覚・身体・MuJoCo統合、FlyBody利用、GPUシミュレーション等の参考実装として利用する。

FlyGym 2.xは旧APIと互換ではないため、依存バージョンを明示して固定する。

## 4. MuJoCo

- Project: https://mujoco.org/
- Repository: https://github.com/google-deepmind/mujoco
- Menagerie: https://github.com/google-deepmind/mujoco_menagerie

身体物理の第一候補。

## 5. MaleCNSアクセス

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

## 6. 今後追加する一次文献

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
- optic flow / looming response
- proprioception / mechanosensation

各実装パラメータについて、「何となく一般的な神経科学モデル」を流用せず、ショウジョウバエで利用可能なデータを優先する。

## 7. 参照時のルール

1. ブログ・解説記事より一次論文・公式データを優先する。
2. 別種の動物由来の値を使う場合は `assumed` または `inferred` として明示する。
3. データセット更新時に自動追従しない。実験単位で版を固定する。
4. 実測コネクトームと、そこから生成した仮想個体の状態を別物として保存する。
5. 外部ソフトウェアに含まれるRL・NN制御器を、明示せずCNSの代わりに使用しない。
