#!/bin/bash

# =================================================================
# CONFIGURATION
# =================================================================

# Fixed Paths
WSI_DIR="/data_64T_3/Dataset/CAMELYON16/images/Tumor"
FEATURE_DIR="/data_64T_3/Raja/MUFASA/2.Attention_Analysis/4.Results_and_plots/Extracted_Coordinates_For_Test_WSI/TRIDENT/Extracted_features/resnet50_1024"
COORDS_DIR="/data_64T_3/Raja/MUFASA/2.Attention_Analysis/4.Results_and_plots/Extracted_Coordinates_For_Test_WSI/TRIDENT/Coordinates"
BASE_OUTPUT_DIR="/data_64T_3/Raja/MUFASA/2.Attention_Analysis/4.Results_and_plots/Extracted_Attention_CAMELYON16"
DEVICE=3

# Model Settings
MODEL_NAME="CLAM_MB_MIL"
MODEL_CONFIG="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/configs/CLAM_MB_MIL.yaml"
CHECKPOINT="/data_64T_3/Raja/MUFASA/1.WSI_Classification/Classification_Logs/CLASSIFICATION/TRIDENT/CAMELYON16/CLAM_MB_MIL/resnet50_1024/time_2026-01-04-15-49_CAMELYON16_CLAM_MB_MIL_seed_2023_resnet50_1024/fold_1/Best_EPOCH_33.pth" 

WSI_LIST=("test_011" "test_021" "test_026" "test_065" "test_068" "test_090" "test_102" "test_108" "test_116" "test_117" "test_121")

# # Fixed Paths
# WSI_DIR="/data_64T_3/Dataset/TCGA_LUSC/images/Tumor_DX1"
# FEATURE_DIR="/data_64T_3/Raja/MUFASA/2.Attention_Analysis/4.Results_and_plots/Extracted_Coordinates_For_Test_WSI/TRIDENT/Extracted_features/resnet50_1024"
# COORDS_DIR="/data_64T_3/Raja/MUFASA/2.Attention_Analysis/4.Results_and_plots/Extracted_Coordinates_For_Test_WSI/TRIDENT/Coordinates"
# BASE_OUTPUT_DIR="/data_64T_3/Raja/MUFASA/2.Attention_Analysis/4.Results_and_plots/Extracted_Attention"
# DEVICE=3

# # Model Settings
# MODEL_NAME="CLAM_MB_MIL"
# MODEL_CONFIG="/home/rajaj/Project/7.WSI_Analysis_Experiments/1.WSI_Classification/configs/CLAM_MB_MIL.yaml"
# CHECKPOINT="/data_64T_3/Raja/MUFASA/1.WSI_Classification/Classification_Logs/CLASSIFICATION/TRIDENT/TCGA_NSCLC/CLAM_MB_MIL/resnet50_1024/time_2026-01-08-02-41_TCGA_NSCLC_CLAM_MB_MIL_seed_42_resnet50_1024/fold_2/Best_EPOCH_144.pth" 

# WSI_LIST=("TCGA-85-7696-01Z-00-DX1.d8756b4c-819f-4a5c-b148-125b8c6b3c27" "TCGA-85-7710-01Z-00-DX1.8bcfe326-1c19-4ebe-879d-83e77739f006" "TCGA-85-7844-01Z-00-DX1.9035209b-75ae-4c9b-8edf-5851af490f29" "TCGA-85-8048-01Z-00-DX1.1a663caa-dff1-4f03-b06b-c8b8b5ce08c6" "TCGA-85-8049-01Z-00-DX1.59554c0f-d04a-41f2-b152-654a848b0443" "TCGA-85-8052-01Z-00-DX1.26b66ae7-b73f-4263-85f7-e82dab5b657b" "TCGA-85-8071-01Z-00-DX1.876f6b0b-f615-43b2-8923-ee21f94568b7" "TCGA-85-8276-01Z-00-DX1.6306624d-e197-4744-a872-0a20413bd9fa" "TCGA-85-8277-01Z-00-DX1.ffad077e-e969-40ba-8504-2f2103911758" "TCGA-85-8288-01Z-00-DX1.0e68535f-a1f3-450c-b8fa-a859f2269da8")

# =================================================================
# EXECUTION LOOP
# =================================================================

# Loop through the array defined above
# "${WSI_LIST[@]}" expands to the items in your list
for WSI_NAME in "${WSI_LIST[@]}"
do
    echo "----------------------------------------------------------------"
    echo "Processing Slide: $WSI_NAME"
    echo "----------------------------------------------------------------"

    # Define specific output folder for this slide
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/TRIDENT"

    python visualize_heatmaps_single_wsi.py \
        --wsi_dir "$WSI_DIR" \
        --feature_dir "$FEATURE_DIR" \
        --coords_dir "$COORDS_DIR" \
        --wsi_name "$WSI_NAME" \
        --model_name "$MODEL_NAME" \
        --model_config "$MODEL_CONFIG" \
        --checkpoint "$CHECKPOINT" \
        --output_dir "$OUTPUT_DIR" \
        --device "$DEVICE"

    echo "Done with $WSI_NAME."
    echo ""
done