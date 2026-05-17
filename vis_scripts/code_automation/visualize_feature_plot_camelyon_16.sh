#!/bin/bash

# =================================================================
# CONFIGURATION
# =================================================================

# 1. Base Paths
INPUT_PATH="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification"
OUTPUT_BASE="/data_64T_3/Raja/MUFASA/VISUALIZATION/CAMELYON_16/FEATURE_PLOTS"

# 2. Model Settings
# Define multiple models here
MODEL_NAMES=("TDA_MIL" "PTC_MIL")

# conda activate wsi_mil_classification
#MODEL_NAMES=("MEAN_MIL" "CLAM_MB_MIL" "DS_MIL" "TRANS_MIL" "DTFD_MIL" "MHIM_MIL" "DGR_MIL" "AC_MIL" "MICO_MIL" "DeepAttn_MIL")

# conda activate mambamil
# MODEL_NAMES=("MAMBA_MIL")

# conda activate transformer
# MODEL_NAMES=("TDA_MIL" "PTC_MIL")

# Pre-processing types
# Define multiple models here
PRE_PROCESSINGS=("CLAM" "TRIDENT" "HISTOLAB" "MUFASA")

# Pre-processing models
# PRE_PROCESSINGS=("CLAM" "TRIDENT" "HISTOLAB" "MUFASA")

# 3. Seeds
SEEDS=(41 1337 2023 3407 9999) 

# =================================================================
# MAIN LOOP
# =================================================================

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for PRE_PROCESSING in "${PRE_PROCESSINGS[@]}"; do
        for SEED in "${SEEDS[@]}"; do
            
            echo "==================================================================="
            echo "Processing: $PRE_PROCESSING | Model: $MODEL_NAME | Seed: $SEED"
            
            # ---------------------------------------------------------
            # 1. DYNAMICALLY FIND CHECKPOINT
            # ---------------------------------------------------------
            
            PARENT_DIR="${INPUT_PATH}/logs/${PRE_PROCESSING}/CAMELYON16/${MODEL_NAME}"
            
            # Look for any folder starting with "resnet50_1024" (handles _set1, etc.)
            RESNET_DIR=$(find "$PARENT_DIR" -maxdepth 1 -type d -name "resnet50_1024*" | head -n 1)

            if [ -z "$RESNET_DIR" ]; then
                 echo "❌ Critical Error: Could not find any 'resnet50_1024*' folder in:"
                 echo "   $PARENT_DIR"
                 continue
            fi

            # Find the specific run folder for this seed
            RUN_FOLDER=$(find "$RESNET_DIR" -maxdepth 1 -type d -name "*seed_${SEED}_*" | head -n 1)

            if [ -z "$RUN_FOLDER" ]; then
                echo "❌ Warning: Run directory not found for ${PRE_PROCESSING} seed ${SEED}"
                continue
            fi

            # Find the .pth file inside fold1 (or fold_1)
            CKPT_PATH=$(find "$RUN_FOLDER" -path "*/fold*" -name "Best_*.pth" | head -n 1)

            if [ -z "$CKPT_PATH" ]; then
                echo "❌ Warning: No .pth checkpoint found inside $RUN_FOLDER"
                continue
            fi

            echo "✅ Found Checkpoint: $CKPT_PATH"

            # ---------------------------------------------------------
            # 2. DYNAMICALLY FIND CSV DATASET
            # ---------------------------------------------------------
            
            DATASET_BASE="${INPUT_PATH}/datasets/CAMELYON16"
            
            # Search for the splits folder dynamically (handles _set1_splits, etc.)
            # Pattern: CAMELYON16_<PREPROC>_resnet50_1024*_splits
            SPLIT_DIR_NAME="CAMELYON16_${PRE_PROCESSING}_resnet50_1024*_splits"
            SPLIT_DIR=$(find "$DATASET_BASE" -maxdepth 1 -type d -name "$SPLIT_DIR_NAME" | head -n 1)

            if [ -z "$SPLIT_DIR" ]; then
                echo "❌ Critical Error: Could not find splits directory matching:"
                echo "   ${DATASET_BASE}/${SPLIT_DIR_NAME}"
                continue
            fi

            CSV_PATH="${SPLIT_DIR}/Camelyon16_binary_class_label_common_split.csv"
            
            if [ ! -f "$CSV_PATH" ]; then
                echo "❌ Critical Error: CSV file not found at:"
                echo "   $CSV_PATH"
                continue
            fi
            
            echo "✅ Found CSV: $CSV_PATH"

            # ---------------------------------------------------------
            # 3. DEFINE OTHER PATHS & EXECUTE
            # ---------------------------------------------------------
            YAML_PATH="${INPUT_PATH}/configs/${MODEL_NAME}.yaml"
            
            # Construct Output Path
            # Outputs will be organized by Model Name folder
            SAVE_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
            SAVE_FILE="${SAVE_DIR}/${PRE_PROCESSING}_seed_${SEED}_feature_plot.png"

            # Ensure output directory exists
            mkdir -p "$SAVE_DIR"

            python draw_feature_map.py \
                --yaml_path "$YAML_PATH" \
                --ckpt_path "$CKPT_PATH" \
                --id2class '{0:"Normal",1:"Tumor"}' \
                --save_path "$SAVE_FILE" \
                --test_dataset_csv "$CSV_PATH" \
                --data_split val \
                --seed "$SEED"

        done
    done
done

echo "All jobs completed."