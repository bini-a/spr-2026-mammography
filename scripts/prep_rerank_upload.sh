#!/usr/bin/env bash
# Prepare a local directory for uploading to Kaggle as a model dataset.
# The directory layout matches what notebook_inference.ipynb expects:
#
#   upload_dir/
#     exp023_bertimbau_dedup/      <- model weights
#     exp015a_bertimbau_seed7/
#     exp015b_bertimbau_seed13/
#     exp015c_bertimbau_seed21/
#     exp034a_023_specialist/
#     exp034b_456_specialist/
#     thresholds.npy               <- from the rerank experiment
#
# Usage:
#   bash scripts/prep_rerank_upload.sh
#   # then upload:
#   kaggle datasets create -p /tmp/exp034_upload --dir-mode zip

set -e

RERANK_EXP="exp034_rerank_a023_0_4_a456_0_2"
OUT_DIR="/tmp/exp034_upload"

MODELS=(
  "exp023_bertimbau_dedup"
  "exp015a_bertimbau_seed7"
  "exp015b_bertimbau_seed13"
  "exp015c_bertimbau_seed21"
  "exp034a_023_specialist"
  "exp034b_456_specialist"
)

echo "Preparing upload directory: $OUT_DIR"
rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

for MODEL in "${MODELS[@]}"; do
  SRC="experiments/$MODEL/model"
  DST="$OUT_DIR/$MODEL"
  if [ ! -d "$SRC" ]; then
    echo "ERROR: model dir not found: $SRC"
    exit 1
  fi
  echo "  Copying $SRC -> $DST"
  cp -r "$SRC" "$DST"
done

echo "  Copying thresholds.npy"
cp "experiments/$RERANK_EXP/thresholds.npy" "$OUT_DIR/thresholds.npy"

echo ""
echo "Upload directory ready: $OUT_DIR"
du -sh "$OUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Create a dataset metadata file (required by Kaggle CLI):"
echo "     kaggle datasets init -p $OUT_DIR"
echo "     # Edit $OUT_DIR/dataset-metadata.json — set title and id"
echo ""
echo "  2. Upload the dataset:"
echo "     kaggle datasets create -p $OUT_DIR --dir-mode zip"
echo ""
echo "  3. Upload the inference notebook to Kaggle:"
echo "     experiments/$RERANK_EXP/notebook_inference.ipynb"
echo ""
echo "  4. In the Kaggle notebook editor:"
echo "     - Add the competition dataset (spr-2026-mammography-report-classification)"
echo "     - Add your uploaded model dataset"
echo "     - Set Accelerator = GPU T4x1, Internet = Off"
echo "     - Run All -> Submit Output"
