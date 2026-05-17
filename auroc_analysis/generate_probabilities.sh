#!/bin/bash

# Configuration
ROOT="/data_64T_3/Raja/MUFASA/1.WSI_Classification/Classification_Logs"
PREPROCESSING=("TRIDENT")  
# "CLAM" "HISTOLAB" "TRIDENT" "MUFASA_SET_1" "MUFASA_SET_1_2_3"
DATASETS=("TCGA_COAD_READ")  
# "TCGA_NSCLC" "STANFORD_793" "TCGA_COAD_READ" "CAMELYON16"
# --- CASE 1: CAMELYON16 (Seed-based Structure) ---
# --- CASE 2: Standard Datasets (Fold-based Structure)  ---
MODELS=("DS_MIL") 
# activate wsi_mil_classification to run "MEAN_MIL" "DeepAttn_MIL" "CLAM_MB_MIL" "DS_MIL" "TRANS_MIL" "DTFD_MIL" "MHIM_MIL" "ADD_MIL" "DGR_MIL" "ILRA_MIL" "AC_MIL" "TDA_MIL"
# activate mambamil to run "MAMBA_MIL" 
FEATURES="resnet50_1024"
# "resnet50_1024" "resnet18" "uni"
GROUP="val"   
# "test" "val" 
SEED=42
DATA_SPLIT_BASE="/data_64T_3/Raja/MUFASA/1.WSI_Classification/Datasets"
OUTPUT_BASE="/data_64T_3/Raja/MUFASA/1.WSI_Classification/AUROC_Analysis/Collected_probabilities"

for P in "${PREPROCESSING[@]}"; do
    for D in "${DATASETS[@]}"; do
        for M in "${MODELS[@]}"; do
            
            BASE_SEARCH_PATH="${ROOT}/${P}/${D}/${M}/${FEATURES}"
            
            if [ ! -d "$BASE_SEARCH_PATH" ]; then
                echo "⚠️ Path not found, skipping: $BASE_SEARCH_PATH"
                continue
            fi

            # --- CASE 1: CAMELYON16 (Seed-based Structure) ---
            if [ "$D" == "CAMELYON16" ]; then
                echo "🧬 Detected CAMELYON16: Switching to Seed-based search."
                
                # Find all timestamped folders for this model
                # e.g., time_2026-01-03..._seed_41_...
                for SEED_FOLDER_PATH in "${BASE_SEARCH_PATH}"/*_seed_*/; do
                    [ -e "$SEED_FOLDER_PATH" ] || continue # Handle empty results
                    
                    # Extract the seed number from the folder name
                    # This regex looks for 'seed_' followed by digits
                    SEED_VAL=$(basename "$SEED_FOLDER_PATH" | grep -oP 'seed_\K[0-9]+')
                    
                    # For Camelyon16, you mentioned they are all in fold_1
                    FOLD_DIR="${SEED_FOLDER_PATH}fold_1"
                    
                    if [ ! -d "$FOLD_DIR" ]; then
                        echo "  ❌ fold_1 missing in $SEED_FOLDER_PATH"
                        continue
                    fi

                    # Find the Best*.pth
                    MODEL_WEIGHT_PATH=$(find "$FOLD_DIR" -name "Best*.pth" | head -n 1)
                    
                    # CSV Path (Fixed name as per your description)
                    CSV_PATH="${DATA_SPLIT_BASE}/${D}/${D}_${P}_${FEATURES}_splits/Camelyon16_binary_class_label_common_split.csv"

                    # Set Output Directory using seed_XX instead of fold_XX
                    LOG_DIR="${OUTPUT_BASE}/${D}/${P}/${M}/seed_${SEED_VAL}"
                    mkdir -p "$LOG_DIR"

                    echo "  🚀 Running Inference | $D | $P | $M | Seed $SEED_VAL"
                    
                    python /home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/test_inference_mil.py \
                        --yaml_path "/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/configs/${M}.yaml" \
                        --model_weight_path "$MODEL_WEIGHT_PATH" \
                        --options \
                            General.seed=$SEED_VAL \
                            Dataset.DATASET_NAME=$D \
                            Dataset.group=$GROUP \
                            Dataset.dataset_csv_path="$CSV_PATH" \
                            Logs.log_root_dir="$LOG_DIR" \
                            General.device=5
                done

            # --- CASE 2: Standard Datasets (Fold-based Structure) ---
            else
                # Get the most recently created folder (the timestamped folder)
                RECENT_FOLDER=$(ls -td "${BASE_SEARCH_PATH}"/*/ | head -1)
                
                echo "📂 Processing latest folder: $RECENT_FOLDER"
    
                # Iterate through fold_1 to fold_5
                for FOLD_NUM in {1..5}; do
                    FOLD_DIR="${RECENT_FOLDER}fold_${FOLD_NUM}"
                    
                    if [ ! -d "$FOLD_DIR" ]; then
                        echo "  ❌ Fold directory missing: $FOLD_DIR"
                        continue
                    fi
    
                    # Find the Best*.pth model weight file
                    MODEL_WEIGHT_PATH=$(find "$FOLD_DIR" -name "Best*.pth" | head -n 1)
    
                    if [ -z "$MODEL_WEIGHT_PATH" ]; then
                        echo "  ❌ No Best*.pth found in $FOLD_DIR"
                        continue
                    fi
    
                    # Construct the CSV path for this specific fold
                    # Note: Adjusting the wildcard to match your pattern: *_{FOLD_NUM}fold_*.csv
                    CSV_PATH=$(find "$DATA_SPLIT_BASE/${D}/${D}_${P}_${FEATURES}_splits" -name "*_${FOLD_NUM}fold_*.csv" | head -n 1)
    
                    # Set Output Directory
                    LOG_DIR="${OUTPUT_BASE}/${D}/${P}/${M}/fold_${FOLD_NUM}"
                    mkdir -p "$LOG_DIR"
    
                    echo "  🚀 Running Inference for | $D | $P | $M | $FEATURES | Fold $FOLD_NUM" |
    
                    # echo "$CSV_PATH"
                    # echo "/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/configs/${M}.yaml"
                    # echo "$MODEL_WEIGHT_PATH"
                    # echo "$LOG_DIR"
    
                    # Execute the python script
                    python /home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/test_inference_mil.py \
                        --yaml_path "/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/configs/${M}.yaml" \
                        --model_weight_path "$MODEL_WEIGHT_PATH" \
                        --options \
                            General.seed=$SEED \
                            Dataset.DATASET_NAME=$D \
                            Dataset.group=$GROUP \
                            Dataset.dataset_csv_path="$CSV_PATH" \
                            Logs.log_root_dir="$LOG_DIR" \
                            General.device=5
                done
            fi 
        done
    done
done

echo "✅ All inference tasks completed."