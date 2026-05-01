#!/usr/bin/env bash
# Prepare the "mammography-exp044-extras" Kaggle dataset.
# Contains: exp041 (seed-99 BERT), exp034a (023 specialist), exp040 (456 specialist).
# The four base models (exp023/015a/015b/015c) stay in mammography-exp998-blend-v1.
#
# Usage:
#   bash scripts/prep_exp044_upload.sh
#   # then:
#   .venv/bin/kaggle datasets create -p /tmp/exp044_extras --dir-mode zip

set -e

OUT_DIR="/tmp/exp044_extras"
MODELS=(
    "exp041_bertimbau_seed99"
    "exp034a_023_specialist"
    "exp040_specialist_456_synth"
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

echo ""
echo "Upload directory ready: $OUT_DIR"
du -sh "$OUT_DIR"
ls -lh "$OUT_DIR"
echo ""

# Write dataset metadata
cat > "$OUT_DIR/dataset-metadata.json" << 'EOF'
{
  "title": "mammography-exp044-extras",
  "id": "biniamgaromsa/mammography-exp044-extras",
  "licenses": [{"name": "CC0-1.0"}]
}
EOF

echo "Metadata written. To upload:"
echo "  .venv/bin/kaggle datasets create -p $OUT_DIR --dir-mode zip"
