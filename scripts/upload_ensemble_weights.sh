#!/usr/bin/env bash
# Prepares and uploads the exp025 ensemble model weights as a Kaggle dataset.
#
# What it does:
#   1. Creates a staging directory with all 4 component model/ dirs + thresholds.npy
#   2. Writes a dataset-metadata.json for the Kaggle API
#   3. Uploads via `kaggle datasets create` (first time) or `version` (updates)
#
# Requirements:
#   - kaggle CLI: pip install kaggle
#   - ~/.kaggle/kaggle.json with your API key (chmod 600)
#
# Usage:
#   bash scripts/upload_ensemble_weights.sh
#   bash scripts/upload_ensemble_weights.sh --update   # push a new version

set -euo pipefail

ENSEMBLE_DIR="experiments/exp025_multiseed_ensemble"
COMPONENTS=(
  "exp023_bertimbau_dedup"
  "exp015a_bertimbau_seed7"
  "exp015b_bertimbau_seed13"
  "exp015c_bertimbau_seed21"
)
STAGING_DIR="/tmp/exp025_upload"
DATASET_TITLE="exp025-multiseed-ensemble"   # shown on Kaggle
DATASET_ID="${DATASET_TITLE}"               # slug used in the notebook

UPDATE=${1:-""}

echo "=== Preparing staging directory ==="
rm -rf "$STAGING_DIR"
mkdir -p "$STAGING_DIR"

for comp in "${COMPONENTS[@]}"; do
  src="experiments/${comp}/model"
  dst="${STAGING_DIR}/${comp}"
  if [ ! -d "$src" ]; then
    echo "ERROR: model dir not found: $src"
    exit 1
  fi
  echo "  Copying $src → $dst"
  cp -r "$src" "$dst"
done

echo "  Copying thresholds.npy"
cp "${ENSEMBLE_DIR}/thresholds.npy" "${STAGING_DIR}/thresholds.npy"

# Write Kaggle dataset metadata
cat > "${STAGING_DIR}/dataset-metadata.json" <<EOF
{
  "title": "${DATASET_TITLE}",
  "id": "$(kaggle config get username 2>/dev/null | tr -d ' \n')/${DATASET_ID}",
  "licenses": [{"name": "other"}]
}
EOF

echo ""
echo "=== Staging directory contents ==="
ls -lh "$STAGING_DIR"

echo ""
if [ "$UPDATE" = "--update" ]; then
  echo "=== Uploading new dataset version ==="
  kaggle datasets version -p "$STAGING_DIR" -m "exp025 ensemble weights update"
else
  echo "=== Creating new Kaggle dataset ==="
  kaggle datasets create -p "$STAGING_DIR" --dir-mode zip
  echo ""
  echo "Dataset created. On Kaggle:"
  echo "  1. Go to your dataset page and confirm it's public (or keep private)"
  echo "  2. Note the dataset slug shown above"
  echo "  3. In the Kaggle notebook, use Add Data → Your Datasets → ${DATASET_TITLE}"
fi

echo ""
echo "Done. Dataset slug for notebook: <your-username>/${DATASET_ID}"
