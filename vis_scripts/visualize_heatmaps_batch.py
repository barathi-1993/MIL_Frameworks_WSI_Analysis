import os
import sys
import argparse
import yaml
import numpy as np
import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

def load_yaml_config(config_file):
    """Load configuration from YAML file"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}

def get_wsi_batch_list(config):
    """
    Determine the list of WSI names to process based on config priority.
    Priority: CSV > TXT > Directory Scan
    """
    wsi_names = []
    source = "Unknown"

    # 1. Check for CSV
    if config.get('wsi_csv') and os.path.exists(config['wsi_csv']):
        print(f"[BATCH] Loading WSI list from CSV: {config['wsi_csv']}")
        try:
            df = pd.read_csv(config['wsi_csv'])
            # Look for common column names
            possible_cols = ['wsi_name', 'slide_id', 'name', 'slide_name']
            col = next((c for c in possible_cols if c in df.columns), None)
            
            if col:
                wsi_names = df[col].astype(str).tolist()
                source = "CSV"
            else:
                print(f"  ! Warning: No valid column found in CSV (expected one of: {possible_cols})")
        except Exception as e:
            print(f"  ! Error reading CSV: {e}")

    # 2. Check for Text File (if no names found yet)
    if not wsi_names and config.get('wsi_list_txt') and os.path.exists(config['wsi_list_txt']):
        print(f"[BATCH] Loading WSI list from Text file: {config['wsi_list_txt']}")
        with open(config['wsi_list_txt'], 'r') as f:
            wsi_names = [line.strip() for line in f if line.strip()]
        source = "TXT"

    # 3. Fallback to Directory Scan
    if not wsi_names and config.get('wsi_dir') and os.path.exists(config['wsi_dir']):
        print(f"[BATCH] Scanning WSI directory: {config['wsi_dir']}")
        supported_ext = ('.svs', '.tif', '.tiff', '.ndpi', '.jpg', '.png')
        files = [f for f in os.listdir(config['wsi_dir']) if f.lower().endswith(supported_ext)]
        # Remove extensions to get IDs
        wsi_names = [os.path.splitext(f)[0] for f in files]
        source = "Directory Scan"

    # Remove duplicates and sort
    wsi_names = sorted(list(set(wsi_names)))
    
    print(f"[BATCH] Found {len(wsi_names)} slides to process (Source: {source})")
    return wsi_names

def validate_inputs(wsi_path, feature_path, coords_path):
    """Quick validation of inputs"""
    errors = []
    
    # Check WSI
    if not Path(wsi_path).exists():
        errors.append(f"WSI not found: {wsi_path}")
    
    # Check features
    if not Path(feature_path).exists():
        errors.append(f"Features not found: {feature_path}")
    
    # Check coordinates
    if not Path(coords_path).exists():
        errors.append(f"Coordinates not found: {coords_path}")
        
    if errors:
        return False, errors
    return True, []

def save_config_yaml(config_dict, output_path):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

def merge_configs(yaml_config, cmd_args):
    """Merge YAML config with command line arguments."""
    merged = yaml_config.copy()
    cmd_dict = vars(cmd_args)
    
    # Defaults that can be overridden
    parser_defaults = {
        'output_dir': None, 'device': 0, 'cmap': 'jet', 
        'alpha': 0.4, 'vis_level': 1, 'patch_size': None, 'config': None
    }
    
    for key, value in cmd_dict.items():
        if key == 'config': continue
        if key in parser_defaults:
            if value != parser_defaults[key]:
                merged[key] = value
        else:
            if value is not None:
                merged[key] = value
    return merged

def process_single_wsi(wsi_name, master_config):
    """Process a single WSI"""
    
    # 1. Construct File Paths
    supported_extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.jpg', '.png']
    wsi_path = None
    
    # Find WSI file with correct extension
    for ext in supported_extensions:
        potential_path = os.path.join(master_config['wsi_dir'], f"{wsi_name}{ext}")
        if os.path.exists(potential_path):
            wsi_path = potential_path
            break
            
    if wsi_path is None:
        return False, f"WSI file not found in {master_config['wsi_dir']}"

    # Construct feature and coordinate paths
    # Note: Adjust naming convention here if your files don't use .pt or _coordinates.npy
    feature_path = os.path.join(master_config['feature_dir'], f"{wsi_name}.pt")
    
    # Try two common coordinate naming conventions
    coords_path_v1 = os.path.join(master_config['coords_dir'], f"{wsi_name}_coordinates.npy")
    coords_path_v2 = os.path.join(master_config['coords_dir'], f"{wsi_name}.npy")
    
    coords_path = coords_path_v1 if os.path.exists(coords_path_v1) else coords_path_v2

    # 2. Validate Files
    is_valid, errors = validate_inputs(wsi_path, feature_path, coords_path)
    if not is_valid:
        return False, "; ".join(errors)

    # 3. Setup Output Directory
    # If a root output dir is defined, create a subfolder for this WSI
    # If output_dir is hardcoded in args, append wsi_name
    base_output = master_config.get('output_dir', 'output')
    # Clean path to avoid output/output nested issues if user passed a full path
    if wsi_name not in base_output:
        wsi_output_dir = os.path.join(base_output, wsi_name)
    else:
        wsi_output_dir = base_output

    os.makedirs(wsi_output_dir, exist_ok=True)

    # 4. Prepare Config for this specific WSI
    wsi_config = {
        'wsi_path': wsi_path,
        'feature_path': feature_path,
        'coords_path': coords_path,
        'model_name': master_config['model_name'],
        'model_config_path': master_config['model_config'],
        'checkpoint_path': master_config['checkpoint'],
        'output_dir': wsi_output_dir,
        'device': master_config['device'],
        'cmap': master_config['cmap'],
        'alpha': master_config['alpha'],
        'vis_level': master_config['vis_level'],
        'thumbnail_size': None,
        'patch_size': master_config['patch_size']
    }

    # 5. Create Temp Config and Run
    timestamp = datetime.now().strftime("%H%M%S")
    temp_config_path = f"temp_vis_{wsi_name}_{timestamp}.yaml"
    
    try:
        with open(temp_config_path, 'w') as f:
            yaml.dump(wsi_config, f, default_flow_style=False, sort_keys=False)
            
        # Run the generation script
        # Using os.system or subprocess. Using string for simplicity as per original script
        cmd = f"python generate_attention_heatmap.py --config {temp_config_path}"
        exit_code = os.system(cmd)
        
        # Save the config used for reference
        save_path = os.path.join(wsi_output_dir, "config.yaml")
        save_config_yaml(wsi_config, save_path)
        
        if exit_code != 0:
            return False, "Visualization script returned error"
            
    except Exception as e:
        return False, str(e)
    finally:
        # Cleanup
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            
    return True, "Success"

def main():
    parser = argparse.ArgumentParser(description='Batch Attention Heatmap Visualization')
    
    # Config file
    parser.add_argument('--config', type=str, default=None, required=True, 
                        help='Path to Batch YAML config file')
    
    # Allow overriding paths
    parser.add_argument('--wsi_dir', type=str, default=None)
    parser.add_argument('--feature_dir', type=str, default=None)
    parser.add_argument('--coords_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    
    # Visualization params
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--cmap', type=str, default='jet')
    parser.add_argument('--alpha', type=float, default=0.4)
    parser.add_argument('--vis_level', type=int, default=1)
    parser.add_argument('--patch_size', type=int, default=None)

    args = parser.parse_args()

    # 1. Load Configuration
    yaml_config = load_yaml_config(args.config)
    config = merge_configs(yaml_config, args)
    
    # 2. Get List of WSIs
    wsi_names = get_wsi_batch_list(config)
    
    if not wsi_names:
        print("✗ No WSIs found to process. Check your CSV, TXT, or WSI directory.")
        return 1
        
    # 3. Batch Processing Loop
    print("\n" + "="*70)
    print(f"STARTING BATCH PROCESSING: {len(wsi_names)} slides")
    print("="*70)
    
    successful = []
    failed = []
    
    # Use tqdm for progress bar
    for i, wsi_name in enumerate(wsi_names):
        print(f"\n[{i+1}/{len(wsi_names)}] Processing: {wsi_name}")
        print("-" * 40)
        
        success, msg = process_single_wsi(wsi_name, config)
        
        if success:
            print(f"  ✓ Done")
            successful.append(wsi_name)
        else:
            print(f"  ✗ Failed: {msg}")
            failed.append((wsi_name, msg))
            
    # 4. Final Summary
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print("="*70)
    print(f"Total: {len(wsi_names)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed:     {len(failed)}")
    
    if failed:
        print("\nFailures:")
        for name, err in failed:
            print(f"  - {name}: {err}")
            
    # Save failure log if any
    if failed:
        log_path = os.path.join(config.get('output_dir', 'output'), 'failure_log.txt')
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            for name, err in failed:
                f.write(f"{name},{err}\n")
        print(f"\nFailure log saved to: {log_path}")

    return 0

if __name__ == '__main__':
    main()