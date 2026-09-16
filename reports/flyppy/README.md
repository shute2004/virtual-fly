# Flyppy reports

`reports/flyppy/` 直下には、現在の状態を参照するための継続更新レポートを置く。

主な現行ファイル:

- `latest.md` / `latest.csv` — 最新 training run
- `population_latest.md` — 最新 population run
- `history.csv` — run 履歴
- `fixed_evaluation_latest.md/.json` — 最新checkpointの固定評価
- `fixed_evaluation_initial_malecns.md/.json` — initial MaleCNS基準

履歴・one-off解析は用途別に分ける。

- `diagnostics/` — causal reach、async bias、motor比較などの診断
- `evaluations/history/` — 過去checkpointの固定評価
- `experiments/` — 独立した比較実験・再現レポート
- `reviews/` — レビュー用資料

新しいone-off解析結果をトップ階層へ増やさず、上記の用途別ディレクトリへ置く。
