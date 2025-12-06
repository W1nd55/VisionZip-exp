#!/usr/bin/env bash
# tools/make_llava_importable.sh
# Helper for VM: installs local model packages in editable mode into the active Python env.
# Usage: source/activate your env first, then run:
#   bash tools/make_llava_importable.sh

set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
PYTHON=${PYTHON:-python3}

echo "Using PYTHON=${PYTHON}"
cd "$ROOT"

pkgs=(models/LLaVA models/VisionZip models/SparseVLMs)
for p in "${pkgs[@]}"; do
  if [ -d "$p" ]; then
    echo "Installing editable: $p"
    "$PYTHON" -m pip install -e "$p" || true
  else
    echo "Skipping missing path: $p"
  fi
done

echo "Done. Verify by running: python3 -c 'import llava; print(llava)'
"
