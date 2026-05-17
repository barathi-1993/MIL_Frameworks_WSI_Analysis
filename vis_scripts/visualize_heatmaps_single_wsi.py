import os
import sys
import subprocess
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path
from datetime import datetime

def load_yaml_config(config_file):
    """Load configuration from YAML file"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            return yaml.safe_load(f)
    return {}

def validate_inputs(wsi_path, feature_path, coords_path):
    """Quick validation of inputs"""
    print("\n[VALIDATION] Checking input files...")
    
    errors = []
    
    # Check WSI
    if not Path(wsi_path).exists():
        errors.append(f"WSI not found: {wsi_path}")
    else:
        print(f"  ✓ WSI: {Path(wsi_path).name}")
    
    # Check features
    if not Path(feature_path).exists():
        errors.append(f"Features not found: {feature_path}")
    else:
        features = torch.load(feature_path)
        if isinstance(features, dict):
            features = features.get('feats', features.get('features'))
        if len(features.shape) == 3:
            features = features.squeeze(0)
        print(f"  ✓ Features: {features.shape}")
    
    # Check coordinates
    if not Path(coords_path).exists():
        errors.append(f"Coordinates not found: {coords_path}")
    else:
        coords = np.load(coords_path)
        print(f"  ✓ Coordinates: {coords.shape}")
        print(f"    Sample: {coords[:3].tolist()}")
        
        # Check correspondence
        if features.shape[0] != coords.shape[0]:
            errors.append(f"Row mismatch: {features.shape[0]} features vs {coords.shape[0]} coords")
        else:
            print(f"  ✓ Row correspondence: {features.shape[0]} tiles")
    
    if errors:
        print("\n  ✗ VALIDATION FAILED:")
        for error in errors:
            print(f"    - {error}")
        return False
    
    print(f"  ✓ All checks passed\n")
    return True

def save_config_yaml(config_dict, output_path):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    print(f"[CONFIG] Saved configuration: {output_path}")

def display_configuration(config_dict):
    """Display all configuration values"""
    print("\n" + "="*70)
    print("CONFIGURATION PARAMETERS")
    print("="*70)
    
    # Group parameters
    groups = {
        'Input Files': ['wsi_path', 'feature_path', 'coords_path'],
        'Model Configuration': ['model_name', 'model_config', 'checkpoint'],
        'Output': ['output_dir'],
        'Visualization': ['cmap', 'alpha', 'vis_level', 'patch_size'],
        'Device': ['device']
    }
    
    for group_name, keys in groups.items():
        print(f"\n{group_name}:")
        for key in keys:
            if key in config_dict:
                value = config_dict[key]
                # # Shorten long paths for display
                # if isinstance(value, str) and len(value) > 80:
                #     display_value = '...' + value[-77:]
                # else:
                #     display_value = value
                print(f"  {key:20s}: {value}")
    
    print("="*70 + "\n")

def merge_configs(yaml_config, cmd_args):
    """
    Merge YAML config with command line arguments.
    Priority: Command line > YAML config > Defaults
    """
    # ============================================
    # UPDATED: Merge logic
    # ============================================
    
    # Start with YAML config
    merged = yaml_config.copy()
    
    # Get command line args that were explicitly provided
    cmd_dict = vars(cmd_args)
    
    # Define which args come from command line (not defaults)
    parser_defaults = {
        'output_dir': None,
        'device': 0,
        'cmap': 'jet',
        'alpha': 0.4,
        'vis_level': 1,
        'patch_size': None,  # Changed from 256 to None
        'config': None
    }
    
    # Override YAML values with command line args (if provided)
    for key, value in cmd_dict.items():
        # Skip if it's the config file argument
        if key == 'config':
            continue
            
        # Check if value was explicitly provided (different from default)
        if key in parser_defaults:
            if value != parser_defaults[key]:
                merged[key] = value
        else:
            # Required args (always from command line)
            if value is not None:
                merged[key] = value

    for key, default_val in parser_defaults.items():
        if key == 'config':
            continue
        if key not in merged:
            merged[key] = default_val
    
    return merged

def run_visualization(config_path):
    """Run the visualization script"""
    print(f"\n[RUNNING] Generating attention heatmap...")
    print("="*70)
    
    # cmd = f"python generate_attention_heatmap.py --config {config_path}"
    # print(f"Command: {cmd}\n")

    cmd = ["python", "generate_attention_heatmap.py", "--config", config_path]
    print(f"Command: {' '.join(cmd)}\n")
    
    # os.system(cmd) 
    subprocess.run(cmd)

def main():
    parser = argparse.ArgumentParser(
        description='Attention Heatmap Visualization with YAML Config Support',
        formatter_class=argparse.RawDescriptionHelpFormatter)

    # ============================================
    # UPDATED: Added config file argument
    # ============================================
    parser.add_argument('--config', type=str, default=None, 
                       help='Path to YAML config file (values can be overridden by command line args)')

    # Input paths (can be overridden from YAML)
    parser.add_argument('--wsi_dir', type=str, default=None, help='Directory containing .svs files')
    parser.add_argument('--feature_dir', type=str, default=None, help='Directory containing .pt feature files')
    parser.add_argument('--coords_dir', type=str, default=None, help='Directory containing .npy coordinate files')
    parser.add_argument('--wsi_name', type=str, default=None, help='WSI name (without extension)')  
    
    # Model (can be overridden from YAML)
    parser.add_argument('--model_name', type=str, default=None, help='Model name')
    parser.add_argument('--model_config', type=str, default=None, help='Path to model config YAML')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory')
    
    # Visualization
    parser.add_argument('--device', type=int, default=0, help='GPU device')
    parser.add_argument('--cmap', type=str, default='jet', help='Colormap')
    parser.add_argument('--alpha', type=float, default=0.4, help='Overlay transparency (0-1)')
    parser.add_argument('--vis_level', type=int, default=1, help='WSI pyramid level')
    parser.add_argument('--patch_size', type=int, default=None, help='Patch size in pixels (None = auto-detect)')
    
    args = parser.parse_args()
    
    # ============================================
    # UPDATED: Load and merge configurations
    # ============================================
    
    # Load YAML config if provided
    yaml_config = load_yaml_config(args.config) if args.config else {}
    
    # Merge YAML and command line args
    config = merge_configs(yaml_config, args)
    
    # Check required parameters
    required_params = ['wsi_dir', 'feature_dir', 'coords_dir', 'wsi_name', 
                      'model_name', 'model_config', 'checkpoint']
    
    missing = [p for p in required_params if p not in config or config[p] is None]
    
    if missing:
        print(f"\n✗ ERROR: Missing required parameters: {', '.join(missing)}")
        print("\nProvide them either via:")
        print("  1. YAML config file (--config config.yaml)")
        print("  2. Command line arguments")
        print("\nRun with --help for details.")
        return 1
    
    # ============================================
    # UPDATED: Construct file paths
    # ============================================
    
    # List of supported extensions
    supported_extensions = ['.svs', '.tif', '.tiff', '.ndpi', '.jpg']
    wsi_path = None

    # Check which file exists
    for ext in supported_extensions:
        potential_path = os.path.join(config['wsi_dir'], f"{config['wsi_name']}{ext}")
        if os.path.exists(potential_path):
            wsi_path = potential_path
            break
    
    if wsi_path is None:
        print(f"\n✗ ERROR: Could not find WSI '{config['wsi_name']}' with extensions {supported_extensions}")
        print(f"   in directory: {config['wsi_dir']}")
        return 1
        
    feature_path = os.path.join(config['feature_dir'], f"{config['wsi_name']}.pt")
    coords_path = os.path.join(config['coords_dir'], f"{config['wsi_name']}_coordinates.npy")
    
    # Update config with constructed paths
    config['wsi_path'] = wsi_path
    config['feature_path'] = feature_path
    config['coords_path'] = coords_path
    
    # Set default output dir if not provided
    config['output_dir'] = f"{config['output_dir']}/{config['model_name']}/{config['wsi_name']}"
 
    # ============================================
    # UPDATED: Display configuration
    # ============================================
    print("="*70)
    print("ATTENTION HEATMAP VISUALIZATION")
    print("="*70)
    
    display_configuration(config)
    
    # Validate inputs
    if not validate_inputs(config['wsi_path'], config['feature_path'], config['coords_path']):
        print("\n✗ Validation failed. Please check your input files.")
        return 1
    
    # ============================================
    # UPDATED: Save configuration to output directory
    # ============================================
    
    # Create output directory
    os.makedirs(config['output_dir'], exist_ok=True)
    
    # Save configuration YAML in output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # config_save_path = os.path.join(config['output_dir'], f"config_{config['wsi_name']}_{timestamp}.yaml")
    config_save_path = os.path.join(config['output_dir'], f"config_{config['wsi_name']}.yaml")
    save_config_yaml(config, config_save_path)
    
    # Also create a temp config for the visualization script
    temp_config_path = f"temp_vis_config_{config['wsi_name']}.yaml"
    
    # Create visualization config
    vis_config = {
        'wsi_path': config['wsi_path'],
        'feature_path': config['feature_path'],
        'coords_path': config['coords_path'],
        'model_name': config['model_name'],
        'model_config_path': config['model_config'],
        'checkpoint_path': config['checkpoint'],
        'output_dir': config['output_dir'],
        'device': config['device'],
        'cmap': config['cmap'],
        'alpha': config['alpha'],
        'vis_level': config['vis_level'],
        'thumbnail_size': None,
        'patch_size': config['patch_size']
    }
    
    with open(temp_config_path, 'w') as f:
        yaml.dump(vis_config, f, default_flow_style=False, sort_keys=False)
    
    # Run visualization
    run_visualization(temp_config_path)
    
    # Clean up temp config
    if os.path.exists(temp_config_path):
        os.remove(temp_config_path)
    
    print("\n" + "="*70)
    print("✓ COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {config['output_dir']}")
    print("\nOutput files:")
    print(f"  - {config['wsi_name']}_attention.npy      (attention scores with coords)")
    print(f"  - {config['wsi_name']}_heatmap.png        (heatmap overlay)")
    print(f"  - {config['wsi_name']}_summary.png        (3-panel visualization)")
    print(f"  - config_{config['wsi_name']}_{timestamp}.yaml  (configuration used)")
    print()
    
    return 0

if __name__ == '__main__':
    exit(main())