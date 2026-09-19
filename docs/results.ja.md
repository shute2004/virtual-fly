# 結果とprovenanceの境界

[English](results.md) · [日本語](results.ja.md) · [简体中文](results.zh-CN.md)

この文書では、どの結果を「現在のcanonical result」と呼んでよいか、どの結果をhistorical development evidenceとしてのみ保持するかを明確にします。

## 1. Canonical experiment v1

Canonical v1は公開時の基準experimentです。科学実行部分はcleanなGit commitに固定されています。

```text
7fa464aad7269d34f46f1171080b51e095d1d811
```

短い結果は [`../canonical/canonical-v1/reference-report.md`](../canonical/canonical-v1/reference-report.md)、完全なprovenanceは [`../canonical/canonical-v1/reference-manifest.json`](../canonical/canonical-v1/reference-manifest.json) にあります。

### 条件

- official MaleCNS v1.0 sourceをfresh取得
- current-semantics MaleCNS snapshotをfresh生成
- 166,700 neurons / 25,582,938 directed edges
- class-DAN semantics: 公開`class=DAN` annotation + dopamine consensus
- body v7 / environment v7
- direct-ray vision、13 rays/ommatidium
- 97-neuron haltere timing subset、gain 0.05、`interaction-load-v2`
- population 2、async shared weights
- boundary-band curriculum
- 6 training episodes
- global-weights-only checkpoint semantics

### Canonicalで観測された結果

- global weight version: v0 → v6
- aggregate neural step: 484
- 厳密に値が変化した保存edge: **2,163,179 / 25,582,938**
- `|Δw| > 1e-7`: 1,461,896 edges
- `|Δw| > 1e-6`: 536,060 edges
- strengthened: 961,611 edges
- weakened: 1,201,568 edges
- maximum `|Δw|`: 0.0018288875

Frozen evaluationではplasticityと課題イベント由来DAN currentを停止しています。initial v0とfinal v6はいずれも1 gate通過後、control step 120で次のgateに衝突しました。

**解釈:** canonical v1は、現行局所可塑性実装によって保存MaleCNS weightが変化した、来歴追跡可能なclosed-loop runを示します。一方、この短いcanonical experimentでは**カテゴリカルな行動改善は観測されていません**。一般化や長期学習安定性も実証していません。

## 2. End-to-end再現確認

2026-09-19に`canonical/canonical-v1/reproduce.sh`を、cleanな`7fa464a` isolated checkoutから最初から最後まで実行しました。

確認済み:

- fresh MaleCNS source download
- fresh snapshot生成
- fresh derived artifact生成
- 記録されたstatic artifact hashが全件byte-for-byte一致
- neural calibrationでstable scales 0.004–0.007から0.005を選択
- canonical 6 episode training完了
- global v0 → v6
- 2,163,179 edgeが厳密に変化しreferenceと一致
- initial/final checkpointを新しいbridge processで再ロード成功
- frozen initial/final evaluationがreference outcomeと一致
- 最終的にpinned scientific worktreeがclean

巨大な検証bundleは一時生成物として扱い、検証後は軽量manifest/reportのみrepository外に保持しました。現在のreproduction scriptは巨大な検証runを既定でrepository外へ出します。

## 3. Historical development results

v240、v960、v966、v1704および関連posting/diagnostic runは、**historical result**として保持します。

これらは開発過程を理解するうえで科学的価値がありますが、canonical v1と同一視できません。[`reproducibility-fixes-2026-09-19.md`](reproducibility-fixes-2026-09-19.md) のprovenance auditには、旧dopamine-source semantics、pre-commit provenance、mixed v966 lineage、現在のfrozen canonical comparisonとは異なるevaluation条件などが記録されています。

historical behaviorを、current class-DAN canonical semanticsで生成された結果のように記述してはいけません。

## 4. Historical Before/After media

公開用historical比較の条件は次です。

- **before:** global weight version v240
- **after:** global weight version v966
- body/environment: v7/v7
- gate 2 center: 13.703125 mm
- evaluation plasticity: off
- before outcome: 1 gate
- after outcome: 2 gates

このmediaはhistorical learned-state comparisonの可視化として有用ですが、canonical behavioral resultではありません。

Gitには小さいrepresentative imageとrelease-asset manifestのみを追跡します。完全なMP4とraw playback JSONはGit外に保持し、GitHub Releaseで明示的にhistoricalとラベル付けして配布します。 [`../release/release-assets-v0.1.0.json`](../release/release-assets-v0.1.0.json) を参照してください。

## 5. README・Release・論文・投稿でのclaimルール

Canonical v1で支持されるclaim:

- 公開MaleCNS connectomeを初期神経構造として使っている
- closed-loopの視覚・機械感覚入力がneural runtimeへ入る
- individual released motor-neuron outputがperipheral modelを介して身体を駆動する
- 課題結果はscalar weight-update targetではなく、選択した実在DAN群への刺激になる
- 現行局所可塑性実装によって保存CNS weightが変化する
- canonical data/provenance pathをend-to-endで再現できる

Canonical v1では支持されないclaim:

- 「canonical v1がFlyppyを解けるように学習した」
- 「canonical v1で行動が改善した」
- 「現在のsimulatorは完全な現実のハエを再現している」
- 「historical v966 behaviorがcurrent canonical semanticsで再現された」
- 「現行modelが一般的な学習能力や長期安定性を実証した」

Historical mediaを示す場合はhistoricalと明示し、この文書またはcanonical reference reportへリンクしてください。
