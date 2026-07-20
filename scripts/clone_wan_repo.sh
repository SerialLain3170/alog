#!/usr/bin/env bash
set -euo pipefail

# Clones the official Wan2.2 repo into /data/shasegawa/vendor (shared host
# storage, not the repo checkout) so both the Docker build and local dev can
# install its pinned torch/flash-attn/xformers stack from its own
# requirements.txt.

VENDOR_DIR="/data/shasegawa/vendor"
REPO_URL="https://github.com/Wan-Video/Wan2.2.git"
DEST="$VENDOR_DIR/Wan2.2"

if [ -d "$DEST" ]; then
  echo "Wan2.2 already cloned at $DEST, pulling latest..."
  git -C "$DEST" pull
else
  git clone "$REPO_URL" "$DEST"
fi

echo "Wan2.2 repo ready at $DEST"
echo "Next: pip install -r $DEST/requirements.txt (inside the worker image/venv)"
