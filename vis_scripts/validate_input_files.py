import argparse
import numpy as np
import torch
from pathlib import Path

def validate_files(feature_path, coords_path):
    """
    Validate that feature and coordinate files are in correct format
    
    Args:
        feature_path: Path to .pt feature file
        coords_path: Path to .npy coordinate file
    
    Returns:
        bool: True if validation passed
    """
    print("="*70)
    print("INPUT FILE VALIDATION")
    print("="*70)
    print(f"Feature file: {feature_path}")
    print(f"Coordinate file: {coords_path}")
    print()
    
    errors = []
    warnings = []
    
    # -------------------------------------------------------------------------
    # Check 1: Files exist
    # -------------------------------------------------------------------------
    print("[CHECK 1] File existence...")
    
    if not Path(feature_path).exists():
        errors.append(f"Feature file not found: {feature_path}")
        print(f"  ✗ Feature file NOT found")
    else:
        print(f"  ✓ Feature file exists")
    
    if not Path(coords_path).exists():
        errors.append(f"Coordinate file not found: {coords_path}")
        print(f"  ✗ Coordinate file NOT found")
    else:
        print(f"  ✓ Coordinate file exists")
    
    if errors:
        print("\n" + "="*70)
        print("VALIDATION FAILED")
        print("="*70)
        for error in errors:
            print(f"  ✗ {error}")
        return False
    
    print()
    
    # -------------------------------------------------------------------------
    # Check 2: Load features
    # -------------------------------------------------------------------------
    print("[CHECK 2] Loading features...")
    
    try:
        features = torch.load(feature_path)
        print(f"  ✓ Successfully loaded .pt file")
        print(f"  ✓ Type: {type(features)}")
        
        # Handle dict format
        if isinstance(features, dict):
            print(f"  ℹ Features stored in dict with keys: {list(features.keys())}")
            
            if 'feats' in features:
                features = features['feats']
                print(f"  ✓ Using 'feats' key")
            elif 'features' in features:
                features = features['features']
                print(f"  ✓ Using 'features' key")
            else:
                errors.append(f"Dict doesn't contain 'feats' or 'features' key. Keys: {list(features.keys())}")
                print(f"  ✗ Expected 'feats' or 'features' key in dict")
        
        # Check if tensor
        if not isinstance(features, torch.Tensor):
            errors.append(f"Features should be torch.Tensor, got {type(features)}")
            print(f"  ✗ Features should be torch.Tensor")
        else:
            print(f"  ✓ Features is torch.Tensor")
        
        # Squeeze if needed
        if len(features.shape) == 3:
            features = features.squeeze(0)
            print(f"  ℹ Squeezed 3D tensor to 2D")
        
        # Check shape
        if len(features.shape) != 2:
            errors.append(f"Features should be 2D (N, feature_dim), got shape {features.shape}")
            print(f"  ✗ Expected 2D shape, got {features.shape}")
        else:
            num_features, feature_dim = features.shape
            print(f"  ✓ Features shape: ({num_features}, {feature_dim})")
            print(f"    - Number of tiles: {num_features}")
            print(f"    - Feature dimension: {feature_dim}")
            
            # Check common feature dimensions
            common_dims = {
                384: "ViT-S",
                512: "CONCH",
                768: "ViT-B",
                1024: "ResNet50/UNI",
                1536: "GigaPath",
                2048: "ResNet101"
            }
            if feature_dim in common_dims:
                print(f"    - Likely encoder: {common_dims[feature_dim]}")
            
    except Exception as e:
        errors.append(f"Failed to load features: {str(e)}")
        print(f"  ✗ Error loading features: {str(e)}")
        features = None
    
    print()
    
    # -------------------------------------------------------------------------
    # Check 3: Load coordinates
    # -------------------------------------------------------------------------
    print("[CHECK 3] Loading coordinates...")
    
    try:
        coords = np.load(coords_path)
        print(f"  ✓ Successfully loaded .npy file")
        print(f"  ✓ Type: {type(coords)}")
        print(f"  ✓ Dtype: {coords.dtype}")
        
        # Check shape
        if len(coords.shape) != 2:
            errors.append(f"Coordinates should be 2D (N, 2), got shape {coords.shape}")
            print(f"  ✗ Expected 2D shape, got {coords.shape}")
        else:
            num_coords, coord_dim = coords.shape
            print(f"  ✓ Coordinates shape: ({num_coords}, {coord_dim})")
            
            if coord_dim != 2:
                errors.append(f"Coordinates should have 2 columns [x, y], got {coord_dim}")
                print(f"  ✗ Expected 2 columns [x, y], got {coord_dim}")
            else:
                print(f"  ✓ Correct format: [x, y] pairs")
                
                # Show sample coordinates
                print(f"\n  Sample coordinates (first 5):")
                for i in range(min(5, len(coords))):
                    print(f"    Row {i}: [{coords[i, 0]}, {coords[i, 1]}]")
                
                # Coordinate statistics
                x_coords = coords[:, 0]
                y_coords = coords[:, 1]
                
                print(f"\n  Coordinate statistics:")
                print(f"    - Total tiles: {num_coords}")
                print(f"    - X range: [{x_coords.min()}, {x_coords.max()}]")
                print(f"    - Y range: [{y_coords.min()}, {y_coords.max()}]")
                
                # Estimate patch size
                unique_x = np.unique(x_coords)
                if len(unique_x) > 1:
                    x_diffs = np.diff(sorted(unique_x))
                    x_diffs = x_diffs[x_diffs > 0]
                    if len(x_diffs) > 0:
                        patch_size = int(np.median(x_diffs))
                        print(f"    - Estimated patch size: {patch_size}px")
                
    except Exception as e:
        errors.append(f"Failed to load coordinates: {str(e)}")
        print(f"  ✗ Error loading coordinates: {str(e)}")
        coords = None
    
    print()
    
    # -------------------------------------------------------------------------
    # Check 4: Row-by-row correspondence
    # -------------------------------------------------------------------------
    print("[CHECK 4] Row-by-row correspondence...")
    
    if features is not None and coords is not None:
        if features.shape[0] == coords.shape[0]:
            print(f"  ✓ Number of rows match: {features.shape[0]}")
            print(f"  ✓ Row i in features.pt corresponds to row i in coordinates.npy")
        else:
            errors.append(f"Row count mismatch: {features.shape[0]} features vs {coords.shape[0]} coordinates")
            print(f"  ✗ Row count mismatch!")
            print(f"    - Features: {features.shape[0]} rows")
            print(f"    - Coordinates: {coords.shape[0]} rows")
    else:
        warnings.append("Cannot check correspondence due to previous errors")
        print(f"  ⚠ Cannot verify correspondence (previous errors)")
    
    print()
    
    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("="*70)
    if errors:
        print("VALIDATION FAILED")
        print("="*70)
        print("\nErrors:")
        for i, error in enumerate(errors, 1):
            print(f"  {i}. {error}")
        
        if warnings:
            print("\nWarnings:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        print("\nPlease fix the errors above before running visualization.")
        return False
    else:
        print("✓ VALIDATION PASSED")
        print("="*70)
        
        if warnings:
            print("\nWarnings:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
            print()
        
        print("\nYour files are correctly formatted!")
        print("\nNext steps:")
        print("  1. Create a visualization config (see attention_vis_config.yaml)")
        print("  2. Run: python vis_scripts/generate_attention_heatmap.py --config your_config.yaml")
        print()
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Validate input files for attention heatmap visualization')
    parser.add_argument('--feature_path', type=str, required=True, help='Path to .pt feature file')
    parser.add_argument('--coords_path',  type=str, required=True, help='Path to .npy coordinate file')
    
    args = parser.parse_args()
    
    success = validate_files(args.feature_path, args.coords_path)
    
    return 0 if success else 1

if __name__ == '__main__':
    exit(main())
