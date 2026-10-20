#!/usr/bin/env bash
set -euo pipefail

CACHE="${PROPRIOCEPTION_EVAL_CACHE:-${PROPRIOCEPTION_DATA:+$PROPRIOCEPTION_DATA/eval_cache}}"
CACHE="${CACHE:-$HOME/.cache/proprioception-evals}"
REPO_URL="https://github.com/JoachimSchaeffer/CIAware-Bench"
COMMIT="ab55b145a1c2036a6ea6825845ee6b146bf24947"
CHECKOUT="$CACHE/ciaware/CIAware-Bench"

mkdir -p "$CACHE/ciaware"
if [[ ! -d "$CHECKOUT/.git" ]]; then
  timeout 600 git clone --quiet "$REPO_URL" "$CHECKOUT"
fi
if [[ "$(git -C "$CHECKOUT" rev-parse HEAD)" != "$COMMIT" ]]; then
  timeout 600 git -C "$CHECKOUT" fetch --quiet origin
  git -C "$CHECKOUT" checkout --quiet "$COMMIT"
fi
echo "== CIAware-Bench at $(git -C "$CHECKOUT" rev-parse --short HEAD) in $CHECKOUT"
echo "== $(grep -c . "$CHECKOUT/cia/environments/essay/topics.txt") essay topics"
