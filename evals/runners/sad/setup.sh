#!/usr/bin/env bash
set -euo pipefail

CACHE="${PROPRIOCEPTION_EVAL_CACHE:-${PROPRIOCEPTION_DATA:+$PROPRIOCEPTION_DATA/eval_cache}}"
CACHE="${CACHE:-$HOME/.cache/proprioception-evals}"
SAD_REPO="https://github.com/LRudL/sad.git"
SAD_COMMIT="dfc5c9831a9bcc5c9a9dbdcaa2955aae983cd1d3"
SAD_ROOT="$CACHE/sad"

mkdir -p "$CACHE"
if [ ! -d "$SAD_ROOT/.git" ]; then
  git clone --quiet "$SAD_REPO" "$SAD_ROOT"
fi
if [ "$(git -C "$SAD_ROOT" rev-parse HEAD)" != "$SAD_COMMIT" ]; then
  git -C "$SAD_ROOT" fetch --quiet origin
  git -C "$SAD_ROOT" checkout --quiet "$SAD_COMMIT"
fi
echo "== sad at $(git -C "$SAD_ROOT" rev-parse --short HEAD) in $SAD_ROOT"

PASSWORD="$(grep -m1 '^PASSWORD=' "$SAD_ROOT/unzip.sh" | cut -d'"' -f2)"
python3 - "$SAD_ROOT" "$PASSWORD" <<'EOF'
import os, sys, zipfile
root, password = sys.argv[1], sys.argv[2].encode()
count = 0
for dirpath, _, files in os.walk(root):
    if ".git" in dirpath:
        continue
    for name in files:
        if not name.endswith(".zip") or name == "evals.zip":
            continue
        zip_path = os.path.join(dirpath, name)
        target = zip_path[:-4]
        if os.path.isdir(target):
            continue
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(path=target, pwd=password)
        count += 1
print(f"== extracted {count} zip archives")
EOF

python3 -c "
import sys; sys.path.insert(0, '$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)')
from evals.sad import corpus
print('== questions ok:', len(corpus.pile('facts/human_defaults', 'facts_human_defaults')), 'facts_human_defaults questions')
"
