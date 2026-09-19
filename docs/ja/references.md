# 科学的参考文献と実装根拠の対応表

[English](../en/references.md) · [日本語](references.md) · [简体中文](../zh-CN/references.md)

`virtual-fly`は、単一の論文をそのまま実装したものではありません。コネクトーム、視覚神経科学、電気生理、運動回路、飛翔筋、身体力学など、複数分野の一次研究と公式データを組み合わせて構築しています。

この文書では、単に参考文献を列挙するのではなく、**各文献をvirtual-flyのどの実装判断の根拠として使ったか**を対応付けています。開発中に参照した全資料を網羅することより、現在の実装に直接関係する主要文献を明確に示すことを優先しています。ここに掲載されている文献が、virtual-fly内のすべての数値パラメータを直接実測したという意味ではありません。推定値・仮定値・較正値は、実装と実験来歴の中で引き続き区別します。

## 現在の実装で主要な根拠としている文献

| 文献 | 状態 | 分野 | virtual-flyで主に使った根拠 |
|---|---|---|---|
| Bates et al. (2026), *Distributed control circuits across a brain-and-cord connectome*, Nature | 査読済み | MaleCNS | 成体オスの脳と腹神経索を一体のCNSコネクトームとして扱うこと、全CNSの接続・注釈 |
| Nern et al. (2025), *Connectome-driven neural inventory of a complete visual system*, Nature | 査読済み | 視覚 | 視葉の構成、laminaやR1-R6の再構築範囲の限界、不足細胞を勝手に補わない判断 |
| Langen et al. (2015), *The Developmental Rules of Neural Superposition in Drosophila*, Cell | 査読済み | 視覚 | neural superposition、近傍個眼のR1-R6とラミナカートリッジの対応 |
| Juusola et al. (2016), *Electrophysiological Method for Recording Intracellular Voltage Responses of Drosophila Photoreceptors and Interneurons to Light Stimuli In Vivo*, JoVE | 査読済み | 視覚・電気生理 | R1-R6とlamina介在ニューロンの光刺激応答、生理学的な局所視覚応答 |
| Haag et al. (2016), *Complementary mechanisms create direction selectivity in the fly*, eLife | 査読済み | 視覚 | 方向選択性を外部分類器で与えず、視葉の局所回路から生じさせる設計 |
| Lesser et al. (2024), *Synaptic architecture of leg and wing premotor control networks in Drosophila*, Nature | 査読済み | 運動制御 | 翼の運動前回路のモジュール構造、運動ニューロン動員、脚と翼の運動前回路の違い |
| Cheong et al. (2026), *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*, eLife | 査読済み・正式版 | 運動制御 | 成体オス腹神経索における下降性入力→運動前回路→運動出力の構成 |
| Ehrhardt et al. (2025), *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster* | プレプリント | 運動制御 | 翼運動前回路の細胞型単位の構成、運動ニューロンと筋の解釈 |
| Teoh et al. (2025), *How tp1, an indirect wing steering muscle, stabilizes Drosophila’s flight* | プレプリント | 飛翔制御 | 間接操舵筋tp1の飛翔安定化への寄与と翼ヒンジ力学 |
| Vaxenburg et al. (2025), *Whole-body physics simulation of fruit fly locomotion*, Nature | 査読済み | 身体・物理 | FlyBodyの全身形状、MuJoCo身体モデル、飛翔・歩行を含む身体力学 |

## 1. MaleCNSと成体オスCNSコネクトーム

### Bates et al. (2026)

**Bates, A. S., Phelps, J. S., Kim, M. et al.** *Distributed control circuits across a brain-and-cord connectome*. Nature 656, 957–970 (2026).

- DOI: https://doi.org/10.1038/s41586-026-10735-w
- 論文: https://www.nature.com/articles/s41586-026-10735-w
- MaleCNS公式サイト: https://male-cns.janelia.org/
- ダウンロード・プログラムからの取得: https://male-cns.janelia.org/download/
- このプロジェクトで使うデータセット: `male-cns:v1.0`
- neuPrint: https://neuprint.janelia.org/
- 論文付属リポジトリ: https://github.com/flyconnectome/2025malecns

成体オスの脳と腹神経索を連続したCNSコネクトームとして扱ううえで、最も根幹となる文献です。`virtual-fly`では、公開MaleCNSを`t = 0`の神経構造として扱い、その後の機能的な重み変化は元データとは分離して保存します。

視葉の網膜対応を解決する際には、MaleCNS公式注釈の`assignedOlHex1` / `assignedOlHex2`も利用します。公式サイトでは、R1-R6から下降性ニューロンへ至る視覚―運動経路の例も公開されています。

- https://male-cns.janelia.org/media/

## 2. 複眼と視葉回路

### Nern et al. (2025)

**Nern, A., Loesche, F., Takemura, S.-y. et al.** *Connectome-driven neural inventory of a complete visual system*. Nature 641, 1225–1237 (2025).

- DOI: https://doi.org/10.1038/s41586-025-08746-0
- 論文: https://www.nature.com/articles/s41586-025-08746-0

視覚系コネクトームの構成と、再構築範囲の限界を解釈するために使っています。特にlaminaやR1-R6の一部が撮像・再構築範囲から欠けることを、人工的に不足細胞を作り足す理由にはしません。公開データに実際に存在する細胞だけを観測済みとして扱います。

### Langen et al. (2015)

**Langen, M., Agi, E. et al.** *The Developmental Rules of Neural Superposition in Drosophila*. Cell 162, 120–133 (2015).

- DOI: https://doi.org/10.1016/j.cell.2015.05.055
- 公開論文: https://pmc.ncbi.nlm.nih.gov/articles/PMC4646663/

neural superpositionの主要文献です。同じ視軸を見る近傍個眼由来のR1-R6が、同じラミナカートリッジへ収束するという構造を踏まえ、1個眼の6細胞を単純平均して1点へ潰さない設計の根拠にしています。

### Juusola et al. (2016)

**Juusola, M., Dau, A., Zheng, L. & Rien, D.** *Electrophysiological Method for Recording Intracellular Voltage Responses of Drosophila Photoreceptors and Interneurons to Light Stimuli In Vivo*. Journal of Visualized Experiments, issue 112, 54142 (2016).

- DOI: https://doi.org/10.3791/54142
- 公開論文: https://pmc.ncbi.nlm.nih.gov/articles/PMC4993232/

R1-R6光受容細胞やlamina介在ニューロンが、局所的な光刺激へどのように応答するかを理解するための電気生理学的資料として利用しています。

### Haag et al. (2016)

**Haag, J., Arenz, A., Serbe, E., Gabbiani, F. & Borst, A.** *Complementary mechanisms create direction selectivity in the fly*. eLife 5, e17421 (2016).

- DOI: https://doi.org/10.7554/eLife.17421
- 公開論文: https://pmc.ncbi.nlm.nih.gov/articles/PMC4978522/

T4/T5の方向選択性が視葉の局所回路上で形成されることを示す主要文献の一つです。このため現在の視覚入力では、CNS外部で「上方向へ動いた」「下方向へ動いた」といった運動ラベルを先に計算して注入しません。

## 3. 翼の運動前回路と運動ニューロン

### Lesser et al. (2024)

**Lesser, E., Azevedo, A. W., Phelps, J. S. et al.** *Synaptic architecture of leg and wing premotor control networks in Drosophila*. Nature 631, 369–377 (2024).

- DOI: https://doi.org/10.1038/s41586-024-07600-z
- 論文: https://www.nature.com/articles/s41586-024-07600-z

翼の運動前回路が運動モジュールとして構成されることや、翼と脚で運動ニューロン動員の構造が異なることの根拠として使っています。`virtual-fly`では、翼運動ニューロン群を平均して行動を選ぶのではなく、個別運動ニューロンから末梢筋へつながる経路を維持します。

### Cheong et al. (2026)

**Cheong, H. S. J., Eichler, K., Stürner, T. et al.** *Organization of circuits linking descending input to motor output in the Drosophila Male Adult Nerve Cord connectome*. eLife, version of record (2026).

- DOI: https://doi.org/10.7554/eLife.96084.3
- 論文: https://elifesciences.org/articles/96084

成体オス腹神経索において、下降性ニューロンの入力が運動前回路を経て運動ニューロンへ至る構成を確認するための主要文献です。

### Ehrhardt et al. (2025・プレプリント)

**Ehrhardt, E., Whitehead, S. C. et al.** *Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster*. bioRxiv preprint, version 3 (2025).

- DOI: https://doi.org/10.1101/2023.05.31.542897
- 公開記録: https://pmc.ncbi.nlm.nih.gov/articles/PMC10312520/

翼運動前回路を細胞型単位で解釈し、個別の翼運動ニューロンと筋の対応を検討する際の補助資料として利用しています。査読済み最終論文ではなく、プレプリントとして区別して扱います。

## 4. 飛翔筋・操舵・全身力学

### Teoh et al. (2025・プレプリント)

**Teoh, H. K., Biswas, D., Leung, A. et al.** *How tp1, an indirect wing steering muscle, stabilizes Drosophila’s flight*. bioRxiv preprint, version 2 (2025).

- DOI: https://doi.org/10.1101/2025.11.02.686144
- 公開記録: https://pmc.ncbi.nlm.nih.gov/articles/PMC12637562/

間接操舵筋tp1が飛翔安定化へ寄与することと、翼ヒンジの機械特性を解釈するために使っています。この文献も査読済み論文とは分け、プレプリントとして扱います。

### Vaxenburg et al. (2025)

**Vaxenburg, R., Siwanowicz, I., Merel, J. et al.** *Whole-body physics simulation of fruit fly locomotion*. Nature 643, 1312–1320 (2025).

- DOI: https://doi.org/10.1038/s41586-025-09029-4
- 論文: https://www.nature.com/articles/s41586-025-09029-4
- FlyBodyリポジトリ: https://github.com/TuragaLab/flybody
- MuJoCo Menagerie内のFlyBody: https://github.com/google-deepmind/mujoco_menagerie/tree/main/flybody

解剖学的なFlyBody全身モデルと、MuJoCoによる身体物理の主要な根拠です。論文内では歩行・飛翔の実演に強化学習も使われていますが、`virtual-fly`ではその方策を神経系の代わりには使わず、身体形状・物理・駆動系を利用します。

## 5. 主要な公式データ・外部ソフトウェア

以下は一次論文の代わりではありませんが、現在の実装で重要な公式データ源・依存先です。

- **MaleCNS** — https://male-cns.janelia.org/
- **neuPrint** — https://neuprint.janelia.org/
- **FlyBody** — https://github.com/TuragaLab/flybody
- **FlyGym / NeuroMechFly文書** — https://neuromechfly.org/
- **MuJoCo** — https://mujoco.org/
- **MuJoCoリポジトリ** — https://github.com/google-deepmind/mujoco
- **MuJoCo Menagerie** — https://github.com/google-deepmind/mujoco_menagerie

FlyGym標準の複眼`Retina`は合成六角格子を使うため、MaleCNS個体におけるbody IDと網膜位置の実測対応そのものとは扱いません。現在の感覚境界では、MaleCNSの網膜対応注釈と公開接続を優先します。

## 6. 文献利用と来歴のルール

1. 解説記事より、一次論文と公式データを優先する。
2. 「どの実装境界をその論文が支えるのか」を明記し、1本の論文を無関係なパラメータ全体の根拠として扱わない。
3. 査読済み論文、プレプリント、公式データ、ソフトウェア文書を区別する。
4. 文献から直接得ていない値は、必要に応じて`inferred`、`assumed`、`calibrated`として区別する。
5. データセットや外部ソフトウェアは、上流更新へ自動追従せず実験ごとに版を固定する。
6. 実測された元データと、時間発展する仮想個体の状態を分離する。
7. 外部ソフトウェアに付属する強化学習方策やニューラルネットワーク制御器を、仮想CNSの代わりに黙って使わない。
