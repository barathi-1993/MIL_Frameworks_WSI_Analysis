# # Using default models in the file
# python attention_score_summary.py \
# --input_folder "/data_64T_3/Raja/MUFASA/ATTENTION_ANALYSIS/Output/CLAM_MB_MIL" \
# --output_path "/data_64T_3/Raja/MUFASA/ATTENTION_ANALYSIS/Output/CLAM_MB_MIL_Results/"

# # For specific models
# python attention_score_summary.py \
# --input_folder "/path/to/input" \
# --output_path "/path/to/output" \
# --models MUFASA CLAM

import pandas as pd
import glob
import os
import argparse 
# ==========================================
# 1. SCORING LOGIC
# ==========================================

def calculate_score(row, model_name):
    """
    Transforms attention scores based on the specific logic provided.
    """
    # Safety checks
    if 'type' not in row.index: return 0.0
    type = row['type']
    
    if model_name not in row.index or f'{model_name}_attention' not in row.index: return 0.0
    
    indicator = row[model_name]
    score = row[f'{model_name}_attention']

    if model_name == 'MUFASA':
        if type == 'relevant':
            # Present: score, Absent: 0.0
            if indicator == 1:
                return score
            else:
                return 0.0
        elif type == 'irrelevant':
            # Present: 1-score, Absent: 0.5
            if indicator == 1:
                return 1.0 - score
            else:
                return 0.95
    else:
        # CLAM, TRIDENT, HISTOLAB logic
        if type == 'relevant':
            # Present: score, Absent: 0.0
            if indicator == 1:
                return score
            else:
                return 0.0
        elif type == 'irrelevant':
            # Present: 1-score, Absent: 0.5
            if indicator == 1:
                return 1.0 - score
            else:
                return 0.95
    return 0.0

# ==========================================
# 2. LEVEL-SPECIFIC GENERATORS
# ==========================================

def generate_tile_csv(input_folder, output_path):
    """
    Combines all CSV files into one large file without any calculation.
    """
    print(f"\n--- Generating Tile Level Results ---")
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    
    all_dfs = []
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            # Optional: Add filename column for tracking
            # df['source_file'] = os.path.basename(file_path)
            all_dfs.append(df)
        except Exception as e:
            print(f"Error reading {os.path.basename(file_path)}: {e}")

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        save_file = os.path.join(output_path, "tile_level.csv")
        combined_df.to_csv(save_file, index=False)
        print(f"Saved: {save_file} ({len(combined_df)} rows)")
    else:
        print("No CSV files found for tile generation.")

def generate_patch_csv(input_folder, output_path, models):
    """
    Calculates transformed scores and averages them per Patch (ROI).
    Returns the combined DataFrame for use in WSI generation.
    """
    print(f"\n--- Generating Patch Level Results ---")
    csv_files = glob.glob(os.path.join(input_folder, '*.csv'))
    all_patch_dfs = []

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)
            
            # Remove duplicate columns if any
            df = df.loc[:, ~df.columns.duplicated()]

            if 'type' not in df.columns or 'patch_id' not in df.columns:
                print(f"Skipping {os.path.basename(file_path)}: Missing required columns")
                continue

            # 1. Apply Scoring Logic
            available_models = []
            for model in models:
                if model in df.columns and f'{model}_attention' in df.columns:
                    # Create new transformed column
                    df[f'{model}_attention'] = df.apply(lambda row: calculate_score(row, model), axis=1)
                    available_models.append(model)

            # 2. Aggregate by Patch ID
            # Define aggregation: First for metadata, Mean for scores
            meta_cols = ['slide_name', 'gridX', 'gridY', 'patch_x', 'patch_y', 'patch_w', 'patch_h']
            agg_dict = {col: 'first' for col in meta_cols if col in df.columns}
            
            for model in available_models:
                agg_dict[f'{model}_attention'] = 'mean'

            patch_df = df.groupby('patch_id', as_index=False).agg(agg_dict)
            
            # Keep consistent column order
            base_cols = ['slide_name', 'patch_id', 'patch_x', 'patch_y', 'gridX', 'gridY', 'patch_w', 'patch_h']
            cols = [c for c in base_cols if c in patch_df.columns] + [f'{m}_attention' for m in available_models]
            
            all_patch_dfs.append(patch_df[cols])

        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")

    if all_patch_dfs:
        combined_patch = pd.concat(all_patch_dfs, ignore_index=True)
        save_file = os.path.join(output_path, "patch_level.csv")
        combined_patch.to_csv(save_file, index=False)
        print(f"Saved: {save_file} ({len(combined_patch)} rows)")
        return combined_patch
    else:
        print("No patch data generated.")
        return pd.DataFrame()

def generate_wsi_csv(patch_df, output_path):
    """
    Takes the Patch-level DataFrame and averages scores per Slide (WSI).
    """
    print(f"\n--- Generating WSI Level Results ---")
    
    if patch_df.empty:
        print("Input patch dataframe is empty. Cannot generate WSI results.")
        return

    if 'slide_name' not in patch_df.columns:
        print("Error: 'slide_name' missing from patch data.")
        return

    # Identify score columns to average
    score_cols = [c for c in patch_df.columns if '_attention' in c]

    # Aggregate by Slide Name
    wsi_df = patch_df.groupby('slide_name')[score_cols].mean().reset_index()

    save_file = os.path.join(output_path, "wsi_level.csv")
    wsi_df.to_csv(save_file, index=False)
    print(f"Saved: {save_file} ({len(wsi_df)} rows)")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    # Initialize Argument Parser
    parser = argparse.ArgumentParser(description="Process WSI CSV files to Tile, Patch, and WSI levels.")
    
    # Add Arguments
    parser.add_argument(
        '--input_folder', 
        type=str, 
        required=True, 
        help='Path to the folder containing input CSV files.'
    )
    parser.add_argument(
        '--output_path', 
        type=str, 
        required=True, 
        help='Path to the output folder where results will be saved.'
    )
    parser.add_argument(
        '--models', 
        nargs='+', 
        default=['CLAM', 'HISTOLAB', 'TRIDENT', 'MUFASA'], 
        help='List of model names to process (default: CLAM HISTOLAB TRIDENT MUFASA).'
    )

    # Parse Arguments
    args = parser.parse_args()

    # Assign to variables
    INPUT_FOLDER = args.input_folder
    OUTPUT_PATH = args.output_path
    MODELS = args.models

    print(f"Input Folder: {INPUT_FOLDER}")
    print(f"Output Path:  {OUTPUT_PATH}")
    print(f"Models:       {MODELS}")
    
    # Ensure output directory exists
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # 1. Tile Level (Raw combination)
    generate_tile_csv(INPUT_FOLDER, OUTPUT_PATH)

    # 2. Patch Level (Apply calculation logic & Aggregate)
    # Pass the models list dynamically
    patch_data = generate_patch_csv(INPUT_FOLDER, OUTPUT_PATH, MODELS)

    # 3. WSI Level (Aggregate Patch results)
    # Check if patch_data is valid before proceeding
    if patch_data is not None and not patch_data.empty:
        generate_wsi_csv(patch_data, OUTPUT_PATH)
    else:
        print("Skipping WSI generation because Patch data is empty.")