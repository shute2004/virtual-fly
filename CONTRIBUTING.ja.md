# virtual-flyへのContributing

[English](CONTRIBUTING.md) · [日本語](CONTRIBUTING.ja.md) · [简体中文](CONTRIBUTING.zh-CN.md)

`virtual-fly`は研究softwareです。実測生物学、文献由来、推定、工学的仮定、calibrationの区別を壊さない変更を歓迎します。

科学・runtime変更の前に以下を読んでください。

- [`README.ja.md`](README.ja.md)
- [`AGENTS.md`](AGENTS.md)
- [`docs/architecture.ja.md`](docs/architecture.ja.md)
- [`docs/results.ja.md`](docs/results.ja.md)
- [`docs/requirements.md`](docs/requirements.md)

## Branch方針

`main`はstable / publication branchです。`dev/flyppy-v3`等のdevelopment branchを使って構いませんが、公開milestoneはdefault branchを古いまま放置せず`main`へ統合します。

統合条件:

- 関連testが通る
- provenance / semantic contractがdocumented
- blocking regressionを隠していない
- 一時generated artifactをsource controlへ混ぜていない
- publication claimがそのlineageで実際に得たevidenceと一致している

## Scientific invariant

明示的なproject-level設計変更なしに以下を導入しません。

- nervous-system learnerとしてのbackpropagation / gradient descent
- Q-learning / policy gradient / actor-critic / 外部learned policy
- 手書きFlyppy攻略policy
- scalar rewardからの直接weight update
- 既知の局所感覚回路を、最終出力だけ似せる外部feature extractor/classifierで置換すること

canonical architectureでの学習は次の経路です。

```text
sensory stimulation
    → neural dynamics
    → behavior
    → neuromodulatory stimulation
    → local plasticity
    → changed nervous system
```

## Provenance label

可能な限り以下を区別します。

- `observed`
- `literature`
- `inferred`
- `assumed`
- `calibrated`

推定・calibrationされたboundaryを直接観測された生物学的事実のように記述しません。

## Historical / canonical result

v240/v960/v966/v1704等のhistorical artifactやdiagnosticはprovenanceのため保持します。current canonical resultとして書き換えません。

公開result claimは [`docs/results.ja.md`](docs/results.ja.md) と [`canonical/canonical-v1/`](canonical/canonical-v1/) に従います。

## 変更単位

1変更1目的を優先します。例:

- MaleCNS data semantics
- neural state stepping
- plasticity
- checkpoint format
- sensory transduction
- motor/peripheral mapping
- body physics
- reproducibility/reporting

神経dynamics、plasticity、neuromodulation、sensory transduction、CNS→body mapping、reinforcement刺激対象、checkpoint semanticsを変更する場合は関連文書も同じ変更で更新します。

## Testing

新しいcomputational core変更には必要最小限の組み合わせで以下を追加します。

- unit test
- deterministic/reference test
- checkpoint round-trip
- semantic contract test
- 必要なbackend parity check

高速化のために科学意味論を黙って変えてはいけません。

## Data / generated artifact

大容量外部datasetや生成artifactをGitへcommitしません。

Git外に置くものの例:

- raw MaleCNS download
- large normalized snapshot
- checkpoint
- trajectory
- rendered MP4
- build cache / profiling trace

代わりにsmall manifest、hash、config、reportを追跡します。

## Documentation language

公開向けdocumentationは英語defaultです。主要な公開入口には日本語・簡体字中国語版が望ましいです。

内部research/development noteは、元contextを保つため日本語のままで構いません。historical recordを見た目の統一だけのため一括翻訳しません。

## Pull request / commit

科学的意味を持つ変更では次を記載します。

- 何を変えたか
- なぜ変えたか
- observed/literature/inferred/assumed/calibratedのどれか
- どの過去experimentと互換性がなくなるか
- どうtestしたか

## License

contributionした独自成果物はrepositoryの [MIT License](LICENSE) で配布されることに同意するものとします。第三者code/data/assetは、利用条件が許可し必要なattribution/noticeを含められる場合のみ追加してください。
