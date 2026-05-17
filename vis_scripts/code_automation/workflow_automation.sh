#!/bin/bash
# =============================================================================
# End-to-End Attention Heatmap Visualization Workflow
# =============================================================================

echo "=============================================================="
echo "Attention Heatmap Visualization - Complete Workflow"
echo "=============================================================="

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Paths (UPDATE THESE!)
WSI_NAME="SHS-14-15532@A8-1"
WSI_PATH="/path/to/WSIs/${WSI_NAME}.svs"
TILE_DIR="/path/to/tiles/${WSI_NAME}"
FEATURE_PATH="/path/to/features/${WSI_NAME}.pt"
COORDS_DIR="/path/to/coordinates"
MODEL_NAME="CLAM_MB_MIL"
MODEL_CONFIG="configs/${MODEL_NAME}.yaml"
CHECKPOINT="logs/CLAM/TCGA_LUAD/${MODEL_NAME}/fold_0/Best_EPOCH_57.pth"
OUTPUT_DIR="logs/visualization/${MODEL_NAME}/${WSI_NAME}"
GPU_DEVICE=0

echo ""
echo "Configuration:"
echo "  WSI: ${WSI_NAME}"
echo "  Model: ${MODEL_NAME}"
echo "  Output: ${OUTPUT_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Step 1: Extract Coordinates (if not already done)
# -----------------------------------------------------------------------------

COORDS_FILE="${COORDS_DIR}/${WSI_NAME}_coordinates.npy"

if [ ! -f "${COORDS_FILE}" ]; then
    echo "[STEP 1] Extracting tile coordinates from filenames..."
    python vis_scripts/extract_tile_coordinates.py \
        --tile_dir "${TILE_DIR}" \
        --wsi_name "${WSI_NAME}" \
        --output_dir "${COORDS_DIR}" \
        --sort_by filename
    echo "✓ Coordinates extracted"
else
    echo "[STEP 1] Coordinates file already exists: ${COORDS_FILE}"
fi

echo ""

# -----------------------------------------------------------------------------
# Step 2: Create Visualization Configuration
# -----------------------------------------------------------------------------

echo "[STEP 2] Creating visualization configuration..."

CONFIG_FILE="vis_configs/${WSI_NAME}_vis_config.yaml"
mkdir -p vis_configs

cat > "${CONFIG_FILE}" << EOF
# Attention Heatmap Visualization Configuration
# Generated automatically for ${WSI_NAME}

# WSI and feature paths
wsi_path: ${WSI_PATH}
feature_path: ${FEATURE_PATH}
coords_path: ${COORDS_FILE}

# Model configuration
model_name: ${MODEL_NAME}
model_config_path: ${MODEL_CONFIG}
checkpoint_path: ${CHECKPOINT}

# Output configuration
output_dir: ${OUTPUT_DIR}

# GPU device
device: ${GPU_DEVICE}

# Visualization parameters
cmap: jet
alpha: 0.4
vis_level: 1
thumbnail_size: [2048, 2048]
patch_size: 512
EOF

echo "✓ Configuration created: ${CONFIG_FILE}"
echo ""

# -----------------------------------------------------------------------------
# Step 3: Generate Attention Heatmap
# -----------------------------------------------------------------------------

echo "[STEP 3] Generating attention heatmap..."
python vis_scripts/generate_attention_heatmap.py --config "${CONFIG_FILE}"
echo ""

# -----------------------------------------------------------------------------
# Step 4: Display Results
# -----------------------------------------------------------------------------

echo "[STEP 4] Results saved to: ${OUTPUT_DIR}"
echo ""
echo "Output files:"
ls -lh "${OUTPUT_DIR}"
echo ""

# -----------------------------------------------------------------------------
# Step 5: Optional - Open Results
# -----------------------------------------------------------------------------

echo "[STEP 5] Opening results (if display available)..."

# Try to open summary image
SUMMARY_IMG="${OUTPUT_DIR}/${WSI_NAME}_summary.png"
if [ -f "${SUMMARY_IMG}" ]; then
    # macOS
    if command -v open &> /dev/null; then
        open "${SUMMARY_IMG}"
    # Linux
    elif command -v xdg-open &> /dev/null; then
        xdg-open "${SUMMARY_IMG}"
    # Windows
    elif command -v start &> /dev/null; then
        start "${SUMMARY_IMG}"
    else
        echo "  ! Cannot auto-open image. Please manually open:"
        echo "    ${SUMMARY_IMG}"
    fi
else
    echo "  ! Summary image not found"
fi

echo ""
echo "=============================================================="
echo "✓ Workflow Complete!"
echo "=============================================================="
echo ""
echo "Next steps:"
echo "  1. Review heatmap: ${OUTPUT_DIR}/${WSI_NAME}_heatmap.png"
echo "  2. Check attention scores: ${OUTPUT_DIR}/${WSI_NAME}_attention.npy"
echo "  3. Analyze top-k patches using the attention scores"
echo ""
