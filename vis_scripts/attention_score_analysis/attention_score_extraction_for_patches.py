# # Minimum required arguments
# python extract_attention.py \
#     --input_path /data/Input_path \
#     --attention_maps_path /data/ATTENTION_MAPS \
#     --output_path /data/Output_path

# # With all options
# python extract_attention.py \
#     --input_path /data/Input_path \
#     --attention_maps_path /data/ATTENTION_MAPS \
#     --output_path /data/Output_path \
#     --model_name CLAM_MB_MIL \
#     --name_column CLAM \
#     --attention_column CLAM_attention

# # Using short flags
# python extract_attention.py \
#     -i /data/Input_path \
#     -a /data/ATTENTION_MAPS \
#     -o /data/Output_path \
#     -m CLAM_MB_MIL \
#     -n selected \
#     -c attention

import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_attention_scores(input_path, attention_maps_path, output_path, 
                            model_name='CLAM_MB_MIL', 
                            name_col='selected', 
                            attention_col='attention_score'):
    """
    Extract attention scores and match with tile coordinates.
    
    Args:
        input_path: Path to folder with WSI_name_index.csv files
        attention_maps_path: Path to ATTENTION_MAPS folder
        output_path: Path to save updated CSV files
        model_name: Model subfolder name (default: 'CLAM_MB_MIL')
        name_col: Name for binary indicator column (default: 'selected')
        attention_col: Name for attention score column (default: 'attention_score')
    """
    
    input_path = Path(input_path)
    attention_maps_path = Path(attention_maps_path)
    output_path = Path(output_path) / model_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all CSV files
    csv_files = sorted(input_path.glob('*_index.csv'))
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in tqdm(csv_files, desc="Processing WSIs"):
        
        # Extract WSI name (remove _index.csv)
        wsi_name = csv_file.stem.replace('_index', '')
        
        # Check if attention folder exists
        attn_folder = attention_maps_path / model_name / wsi_name
        attn_file = attn_folder / f"{wsi_name}_attention.npy"
        
        # Output file path
        output_file = output_path / f"{wsi_name}_index_updated.csv"
        
        # Check if output file already exists
        if output_file.exists():
            df = pd.read_csv(output_file)
            
            # Check if columns already exist
            if name_col in df.columns and attention_col in df.columns:
                print(f"⚠️  {wsi_name}: Columns '{name_col}' and '{attention_col}' already exist. Updating values...")
            elif name_col in df.columns or attention_col in df.columns:
                print(f"ℹ️  {wsi_name}: One column exists. Adding missing column(s)...")
            else:
                print(f"ℹ️  {wsi_name}: Adding new columns '{name_col}' and '{attention_col}'...")
        else:
            # Read input CSV for first time
            df = pd.read_csv(csv_file)
        
        if attn_folder.exists() and attn_file.exists():
            # Load attention data: [x, y, attention]
            attn_data = np.load(attn_file)
            
            # Create coordinate lookup dict: (x, y) -> attention_value
            coord_to_attn = {(int(row[0]), int(row[1])): row[2] 
                            for row in attn_data}
            
            # Initialize or update columns
            df[name_col] = 0
            df[attention_col] = 0.0
            
            for idx, row in df.iterrows():
                coord = (int(row['tile_x']), int(row['tile_y']))
                if coord in coord_to_attn:
                    df.at[idx, name_col] = 1
                    df.at[idx, attention_col] = coord_to_attn[coord]
            
            matched = df[name_col].sum()
            print(f"✅ {wsi_name}: Matched {matched}/{len(df)} tiles")
        
        else:
            # No attention file found - set columns to 0
            df[name_col] = 0
            df[attention_col] = 0.0
            print(f"⚠️  {wsi_name}: No attention file found")
        
        # Save output
        df.to_csv(output_file, index=False)
    
    print(f"\n✅ Done! Output saved to: {output_path}")


# MAIN EXECUTION
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description='Extract attention scores and match with tile coordinates',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--input_path', '-i', type=str, required=True,
                        help='Path to folder containing WSI_name_index.csv files')
    
    parser.add_argument('--attention_maps_path', '-a', type=str, required=True,
                        help='Path to ATTENTION_MAPS folder')
    
    parser.add_argument('--output_path', '-o', type=str, required=True,
                        help='Path to save updated CSV files')
    
    parser.add_argument('--model_name', '-m', type=str, default='CLAM_MB_MIL',
                        help='Model subfolder name')
    
    parser.add_argument('--name_column', '-n', type=str, default='selected',
                        help='Binary indicator column name')
    
    parser.add_argument('--attention_column', '-c', type=str, default='attention',
                        help='Attention score column name')
    
    args = parser.parse_args()
    
    extract_attention_scores(
        input_path=args.input_path,
        attention_maps_path=args.attention_maps_path,
        output_path=args.output_path,
        model_name=args.model_name,
        name_col=args.name_column,
        attention_col=args.attention_column
    )