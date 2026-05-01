#!/usr/bin/env bash
# Prepare a local directory for uploading to Kaggle as a model dataset.
# Layout:
#   upload_dir/
#     exp023_bertimbau_dedup/
#     exp015a_bertimbau_seed7/
#     exp015b_bertimbau_seed13/
#     exp015c_bertimbau_seed21/
#     exp030_bertimbau_synthetic_rare/
#     exp031_bertimbau_synthetic_c56/
#     thresholds.npy               <- from exp998_blend_base

set -e

OUT_DIR="/tmp/exp998_upload"

MODELS=(
  "exp023_bertimbau_dedup"
  "exp015a_bertimbau_seed7"
  "exp015b_bertimbau_seed13"
  "exp015c_bertimbau_seed21"
  "exp030_bertimbau_synthetic_rare"
  "exp031_bertimbau_synthetic_c56"
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

echo "  Copying thresholds.npy from exp998_blend_base"
cp "experiments/exp998_blend_base/thresholds.npy" "$OUT_DIR/thresholds.npy"

echo ""
echo "Upload directory ready: $OUT_DIR"
du -sh "$OUT_DIR"
echo ""
echo "Next steps:"
echo "  1. Init dataset metadata:"
echo "     kaggle datasets init -p $OUT_DIR"
echo "     # Edit $OUT_DIR/dataset-metadata.json — set title and id"
echo ""
echo "  2. Upload the dataset:"
echo "     KAGGLE_API_TOKEN=\$KAGGLE_TOKEN kaggle datasets create -p $OUT_DIR --dir-mode zip"
echo ""
echo "  3. Upload the inference notebook:"
echo "     experiments/exp998_blend_base/notebook_inference.ipynb"
