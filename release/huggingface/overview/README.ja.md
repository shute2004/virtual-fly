# virtual-fly

[English](README.md) · [日本語](README.ja.md) · [简体中文](README.zh-CN.md)

`virtual-fly`は、成体オスのショウジョウバエの公開MaleCNSコネクトームを初期状態として、神経活動・神経修飾・局所シナプス可塑性を時間発展させ、FlyBody / MuJoCoの身体と物理環境に閉ループ接続する研究プロジェクトです。

このHugging Faceリポジトリでは、`virtual-fly`の概要と、初回公開に向けて準備しているチェックポイントリポジトリの構成をまとめています。ソースコード、科学的な説明、canonical実験のmanifest、一連の再現スクリプトはGitHubにあります。

- GitHub: https://github.com/shute2004/virtual-fly
- 英語文書: https://github.com/shute2004/virtual-fly/tree/main/docs/en
- 日本語README: https://github.com/shute2004/virtual-fly/blob/main/README.ja.md
- 简体中文README: https://github.com/shute2004/virtual-fly/blob/main/README.zh-CN.md

## チェックポイントの配布構成

公開チェックポイントには、明確に異なる2つの系統があります。**動画用のBefore / Afterをcanonical v1の初期・学習後チェックポイントとして扱わないでください。**

### Canonical v1

canonical系の公開日は2026年9月20日です。

| チェックポイント | リポジトリ | 意味 |
|---|---|---|
| 初期 | [`shute2004/virtual-fly-initial-20260920`](https://huggingface.co/shute2004/virtual-fly-initial-20260920) | canonical v1の初期チェックポイント。global weight version 0 |
| 学習後 | [`shute2004/virtual-fly-trained-20260920`](https://huggingface.co/shute2004/virtual-fly-trained-20260920) | canonical v1で6エピソードの学習処理を経たチェックポイント。global weight version 6 |

記録済みのcanonical v1実験は、実際にはGit commit `7fa464aad7269d34f46f1171080b51e095d1d811`で実行しました。公開前のprivacy-onlyな履歴書き換え後、公開用の科学コード相当commitは`f9c86c904d67ff974f3c43d37aab3619bc93fc1b`です。変更はhistorical absolute path内のローカルOSユーザー名だけで、実行コードやcanonical条件は変わっていません。この短いcanonical実験では、保存された25,582,938本の辺のうち2,163,179本で重みが変化しました。一方、可塑性とタスク由来のDAN刺激を無効化した固定評価では、初期・学習後とも同じ分類上の結果でした。

したがってcanonical v1が示しているのは、**現行の局所可塑性の意味づけのもとで保存重みが実際に変化したこと**であり、明確な行動改善や一般化を実証したという主張ではありません。

### 過去系統のBefore / After動画

| チェックポイント | リポジトリ | 来歴上の役割 |
|---|---|---|
| Before | [`shute2004/virtual-fly-video-before`](https://huggingface.co/shute2004/virtual-fly-video-before) | 公開したBefore / After比較動画のBefore側で実際に使用した過去系統のチェックポイント |
| After | [`shute2004/virtual-fly-video-after`](https://huggingface.co/shute2004/virtual-fly-video-after) | 同じ動画のAfter側で実際に使用した過去系統のチェックポイント |

これらの内部開発版は来歴情報として`v240`と`v966`ですが、公開リポジトリ名には出しません。Before側のv240は未学習の初期状態ではなく、すでに240エピソードの過去学習を経ています。再生はホールドアウト評価ではなく、可塑性を停止した状態でも課題イベントによるPAM刺激は有効でした。v966も複数の過去条件が混在した系統です。したがって、これらは**canonical v1の証拠ではありません**。

## Hugging Faceに置くもの

各チェックポイント用リポジトリには、チェックポイント本体と、必要最小限の来歴情報、ハッシュ値、帰属情報、モデルカードを置きます。研究用作業環境全体を複製するものではありません。

原則として置かないもの:

- MaleCNSの生データ
- 正規化済みMaleCNSスナップショット一式
- FlyBody / FlyGym / MuJoCoのソースコードや身体資産
- 軌跡データ一式
- 描画フレーム一式
- 今回の公開対象と無関係な開発途中のチェックポイント
- ビルドキャッシュや仮想環境

上流データと、そこから生成する静的データについては、MaleCNS公式配布元とGitHubのcanonical再現スクリプトを取得・再生成の正規経路とします。

## ライセンスと帰属

`virtual-fly`独自のソースコードと文書はMIT Licenseで公開します。チェックポイントの重みは、MaleCNS公式サイトがCC BY 4.0で配布している`male-cns:v1.0`を基礎として生成した派生データです。そのため、チェックポイント用リポジトリは`cc-by-4.0`として明示し、MaleCNSへの帰属情報を付けます。

FlyBody、FlyGym、MuJoCoはそれぞれ上流のライセンスに従い、利便性だけを理由にチェックポイント用リポジトリへ複製しません。

詳しい境界はGitHubリポジトリの`THIRD_PARTY_NOTICES.md`を参照してください。

## 再現

現在のcanonical実験とソースレベルの完全な来歴については、GitHub側を使用してください。

```bash
git clone https://github.com/shute2004/virtual-fly.git
cd virtual-fly
uv sync --frozen
bash canonical/canonical-v1/reproduce.sh
```

Hugging Faceの各チェックポイントリポジトリは、チェックポイントを配布するためのものです。canonical v1の科学的な定義は、GitHub上のcanonicalパッケージを基準とします。
