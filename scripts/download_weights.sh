#!/usr/bin/env bash
set -euo pipefail

# Downloads Wan2.2-Animate-14B checkpoint weights into /data/shasegawa/vendor
# (shared host storage, ~72.4GB).
# Requires: pip install "huggingface_hub[cli]"

VENDOR_DIR="/data/shasegawa/vendor"
DEST="$VENDOR_DIR/Wan2.2-Animate-14B"

mkdir -p "$DEST"
huggingface-cli download Wan-AI/Wan2.2-Animate-14B --local-dir "$DEST"

echo "Weights downloaded to $DEST"
