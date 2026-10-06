#!/usr/bin/env bash
# Refit the blend (baseball_stats/blend.json) on every historical price plus the live
# log's closing lines. Rerun after any Strikeouts or Hitters model change merges: the
# weights depend on the model, and the historical model chances are recomputed with the
# current code (walk-forward). Needs data/processed up to date (see CLAUDE.md).
#
#   scripts/refit_blend.sh [weights.json] [report.json]
set -euo pipefail
weights=${1:-baseball_stats/blend.json}
report=${2:-docs/edge_refit.json}
dir=$(mktemp -d)
trap 'rm -rf "$dir"' EXIT
git fetch -q origin odds-history odds-log
mkdir "$dir/prices"
# The root holds the one-hour snapshots the blend is fitted on; other times sit in m<minutes>/.
git archive origin/odds-history | tar -x -C "$dir/prices"
git show origin/odds-log:prop_snapshots.parquet > "$dir/prop_snapshots.parquet"
python scripts/fit_blend.py --fit "$dir/prices" --live "$dir/prop_snapshots.parquet" \
  --weights "$weights" --report "$report"
