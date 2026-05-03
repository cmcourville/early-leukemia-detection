#!/bin/bash
# models/cytodiffusion/data_prep/download_cytodata.sh
#
# Download the CytoData labeled subset from EBI BioStudies (S-BSST2156).
#
# BEFORE RUNNING:
#   The dataset requires a data-access agreement.
#   1. Visit https://www.ebi.ac.uk/biostudies/studies/S-BSST2156
#   2. Accept the terms of use
#   3. Then run this script to download via FTP
#
# Usage:
#   bash models/cytodiffusion/data_prep/download_cytodata.sh [DEST_DIR]
#
#   DEST_DIR defaults to $HOME/data/cytodata_raw
#
# After download, run prepare_cytodata.py to create the ImageFolder layout.

set -e

DEST_DIR="${1:-$HOME/data/cytodata_raw}"
EBI_BASE="https://ftp.ebi.ac.uk/biostudies/fire/S-BSST/156/S-BSST2156/Files"

echo "[download] Destination: $DEST_DIR"
mkdir -p "$DEST_DIR"

# ── Labeled annotations (CSV with image paths, cell types, confidence) ─────────
echo "[download] Fetching label files..."
wget -q --show-progress -P "$DEST_DIR" \
    "$EBI_BASE/labels/train_labels.csv" \
    "$EBI_BASE/labels/val_labels.csv"   \
    "$EBI_BASE/labels/test_labels.csv"  \
    || echo "[download] WARNING: individual split CSVs not found — trying combined labels..."

# Fallback: some releases ship a single labels.csv instead of split files
wget -q --show-progress -P "$DEST_DIR" \
    "$EBI_BASE/labels/labels.csv" 2>/dev/null || true

# ── Single-cell images (labeled subset only — ~5K images, ~1 GB) ───────────────
# The labeled subset images are stored under data/labeled/ in the EBI archive.
echo "[download] Fetching labeled cell images..."
wget -q --show-progress -r --no-parent --no-host-directories \
    --cut-dirs=6 -P "$DEST_DIR/images" \
    "$EBI_BASE/data/labeled/"          \
    || {
        echo "[download] INFO: directory listing not available."
        echo "[download] Trying tar archive download..."
        wget -q --show-progress -P "$DEST_DIR" \
            "$EBI_BASE/data/labeled_cells.tar.gz" || true
    }

# Unpack if a tar was downloaded
if [ -f "$DEST_DIR/labeled_cells.tar.gz" ]; then
    echo "[download] Extracting archive..."
    tar -xzf "$DEST_DIR/labeled_cells.tar.gz" -C "$DEST_DIR/images/"
fi

echo ""
echo "[download] Done. Raw files at: $DEST_DIR"
echo ""
echo "Next step — create ImageFolder layout:"
echo "  python models/cytodiffusion/data_prep/prepare_cytodata.py \\"
echo "    --raw_dir  $DEST_DIR \\"
echo "    --out_dir  \$HOME/data/cytodata"
