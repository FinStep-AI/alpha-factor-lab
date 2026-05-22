#!/usr/bin/env bash
# include_upstream.sh  ←  上游合并脚本（stub）
# 把 permissionPack 里的三方 receptor 片段拼进上游 repo，纯合并不做路由假设。
set -euo pipefail

REPO="${1:-.}"
PACK_DIR="$REPO/permissionPack"
MERGE_LOG="${2:-$PACK_DIR/receptor_merge.log}"

if [ ! -d "$PACK_DIR" ]; then
  echo "[include_upstream] pack dir not found: $PACK_DIR" >&2; exit 1
fi

> "$MERGE_LOG"
for f in "$PACK_DIR"/*; do
  [ -f "$f" ] || continue
  echo "[include_upstream] » $f" >> "$MERGE_LOG"
  cat  "$f"                       >> "$MERGE_LOG"
  echo ""                          >> "$MERGE_LOG"
done

echo "[include_upstream] wrote $MERGE_LOG ($(wc -l < "$MERGE_LOG") lines)"
