#!/bin/bash

# =================================================================
# CONFIGURATION
# =================================================================

# 1. Base Paths
INPUT_PATH="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification"
OUTPUT_BASE="/data_64T_3/Raja/MUFASA/VISUALIZATION/TCGA_NSCLC/FEATURE_PLOTS"

# 2. Model Settings
# conda activate wsi_mil_classification
# MODEL_NAMES=("MEAN_MIL" "CLAM_MB_MIL" "DS_MIL" "TRANS_MIL" "DTFD_MIL" "MHIM_MIL" "DGR_MIL" "AC_MIL" "MICO_MIL" "DeepAttn_MIL")

# conda activate mambamil
# MODEL_NAMES=("MAMBA_MIL")

# conda activate transformer
MODEL_NAMES=("TDA_MIL" "PTC_MIL")

# Pre-processing types 
PRE_PROCESSINGS=("CLAM" "TRIDENT" "HISTOLAB" "MUFASA")

# 3. Folds (1 to 5)
FOLDS=(1 2 3 4 5)

# =================================================================
# MAIN LOOP
# =================================================================

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for PRE_PROCESSING in "${PRE_PROCESSINGS[@]}"; do
        for FOLD_NUM in "${FOLDS[@]}"; do
            
            echo "==================================================================="
            echo "Processing: $PRE_PROCESSING | Model: $MODEL_NAME | Fold: $FOLD_NUM"
            
            # ---------------------------------------------------------
            # 1. DYNAMICALLY FIND CHECKPOINT
            # ---------------------------------------------------------
            
            # Path to the logs directory for this model/preprocessing
            PARENT_DIR="${INPUT_PATH}/logs/${PRE_PROCESSING}/TCGA_NSCLC/${MODEL_NAME}"
            
            # Find the 'resnet50_1024' folder (handles _set1, etc.)
            RESNET_DIR=$(find "$PARENT_DIR" -maxdepth 1 -type d -name "resnet50_1024*" | head -n 1)

            if [ -z "$RESNET_DIR" ]; then
                 echo "❌ Critical Error: Could not find any 'resnet50_1024*' folder in:"
                 echo "   $PARENT_DIR"
                 continue
            fi

            # Find the timestamped run folder
            # Assuming there is only one run folder or taking the first one found.
            # If you have specific seeds per fold, you might need to adjust this.
            # Here we grab the first directory that looks like a run.
            RUN_FOLDER=$(find "$RESNET_DIR" -maxdepth 1 -type d -name "time_*" | head -n 1)

            if [ -z "$RUN_FOLDER" ]; then
                echo "❌ Warning: Run directory not found inside $RESNET_DIR"
                continue
            fi

            # Find the .pth file inside fold{N} or fold_{N}
            # This handles 'fold1' vs 'fold_1'
            CKPT_PATH=$(find "$RUN_FOLDER" -path "*/fold_*${FOLD_NUM}/Best_*.pth" | head -n 1)

            if [ -z "$CKPT_PATH" ]; then
                echo "❌ Warning: No .pth checkpoint found for fold ${FOLD_NUM} in $RUN_FOLDER"
                continue
            fi

            echo "✅ Found Checkpoint: $CKPT_PATH"

            # ---------------------------------------------------------
            # 2. DYNAMICALLY FIND CSV DATASET FOR THIS FOLD
            # ---------------------------------------------------------
            
            DATASET_BASE="${INPUT_PATH}/datasets/TCGA_NSCLC"
            
            # Search for the splits folder
            SPLIT_DIR_NAME="TCGA_NSCLC_${PRE_PROCESSING}_resnet50_1024*_splits"
            SPLIT_DIR=$(find "$DATASET_BASE" -maxdepth 1 -type d -name "$SPLIT_DIR_NAME" | head -n 1)

            if [ -z "$SPLIT_DIR" ]; then
                echo "❌ Critical Error: Could not find splits directory matching:"
                echo "   ${DATASET_BASE}/${SPLIT_DIR_NAME}"
                continue
            fi

            # Find the CSV file corresponding to the current fold number
            # Matches patterns like 'splits_1.csv', 'split_1.csv', or just '*1*.csv'
            CSV_PATH=$(find "$SPLIT_DIR" -maxdepth 1 -type f -name "*${FOLD_NUM}*.csv" | head -n 1)
            
            if [ -z "$CSV_PATH" ]; then
                echo "❌ Critical Error: CSV file for fold ${FOLD_NUM} not found in:"
                echo "   $SPLIT_DIR"
                continue
            fi
            
            echo "✅ Found CSV: $CSV_PATH"

            # ---------------------------------------------------------
            # 3. DEFINE OUTPUT PATH & EXECUTE
            # ---------------------------------------------------------
            YAML_PATH="${INPUT_PATH}/configs/${MODEL_NAME}.yaml"
            
            # Output path: .../MODEL_NAME/PREPROCESSING_foldX_feature_plot.png
            SAVE_DIR="${OUTPUT_BASE}/${MODEL_NAME}"
            SAVE_FILE="${SAVE_DIR}/${PRE_PROCESSING}_fold${FOLD_NUM}_feature_plot.png"

            mkdir -p "$SAVE_DIR"

            python ../draw_feature_map.py \
                --yaml_path "$YAML_PATH" \
                --ckpt_path "$CKPT_PATH" \
                --id2class '{0:"TCGA_LUAD",1:"TCGA_LUSC"}' \
                --save_path "$SAVE_FILE" \
                --test_dataset_csv "$CSV_PATH" \
                --data_split val \
                --seed 42

        done
    done
done

echo "All jobs completed."