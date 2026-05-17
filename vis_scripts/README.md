# Quick Reference Card - Your File Format

## Your Data Structure
```
/data/
├── WSIs/WSI_NAME.svs                    # Whole slide images
├── features/WSI_NAME.pt                 # Features: (N, feature_dim)
└── coordinates/WSI_NAME_coordinates.npy # Coords: (N, 2) [[x,y], ...]
This example shows how to use the visualization system with your exact file format:
- .svs WSI files
- .pt feature files (row-by-row features)
- .npy coordinate files (row-by-row [x, y] pairs)

Your file structure:
--------------------
/data/
  ├── WSIs/
  │   ├── TCGA-A1-A0SB.svs
  │   ├── TCGA-A1-A0SD.svs
  │   └── TCGA-A1-A0SE.svs
  ├── features/
  │   ├── TCGA-A1-A0SB.pt          # Shape: (N, feature_dim)
  │   ├── TCGA-A1-A0SD.pt
  │   └── TCGA-A1-A0SE.pt
  └── coordinates/
      ├── TCGA-A1-A0SB_coordinates.npy  # Shape: (N, 2) - [[x, y], ...]
      ├── TCGA-A1-A0SD_coordinates.npy
      └── TCGA-A1-A0SE_coordinates.npy

Row-by-row correspondence:
---------------------------
features.pt row i  <->  coordinates.npy row i  <->  same tile

Example coordinates.npy content:
[[9984, 7680],
 [9984, 7936],
 [9984, 8192],
 ...]

```

## Three Essential Commands

### 1️⃣ Validate (Recommended First Step)
```bash
python vis_scripts/validate_input_files.py \
    --feature_path /data/features/WSI_NAME.pt \
    --coords_path /data/coordinates/WSI_NAME_coordinates.npy
```

### 2️⃣ Single WSI Visualization
```bash
python complete_example.py \
    --wsi_dir /data/WSIs \
    --feature_dir /data/features \
    --coords_dir /data/coordinates \
    --wsi_name WSI_NAME \
    --model_name CLAM_MB_MIL \
    --model_config configs/CLAM_MB_MIL.yaml \
    --checkpoint logs/Best.pth
```

### 3️⃣ Batch Processing
```bash
# Create wsi_list.csv with WSI names
python vis_scripts/batch_generate_heatmaps.py \
    --config vis_configs/batch_config.yaml
```

## Output Files
```
logs/visualization/MODEL/WSI_NAME/
├── WSI_NAME_attention.npy      # (N, 3): [x, y, attention]
├── WSI_NAME_heatmap.png        # Heatmap overlay
└── WSI_NAME_summary.png        # 3-panel visualization
```

## Load Results
```python
import numpy as np

data = np.load('WSI_NAME_attention.npy')
x_coords = data[:, 0]      # X pixel coordinates
y_coords = data[:, 1]      # Y pixel coordinates
attention = data[:, 2]     # Attention scores

# Top-100 patches
top_100_idx = np.argsort(attention)[-100:]
top_coords = data[top_100_idx, :2]
```

## Supported Models
- CLAM_MB_MIL - Multi-branch with instance learning
- AB_MIL - Attention-based pooling
- MAMBA_MIL - Mamba state space model
- BiMAMBA_MIL - Bidirectional Mamba
- SRMAMBA_MIL - Shuffled residual Mamba
- TDA_MIL - Top-down attention

## File Requirements

✅ **Features (.pt)**
- Shape: `(N, feature_dim)`
- Format: `torch.Tensor` or dict with `'feats'`/`'features'` key
- Row i = feature for tile i

✅ **Coordinates (.npy)**
- Shape: `(N, 2)`
- Format: `[[x1, y1], [x2, y2], ...]`
- Row i = [x, y] for tile i

✅ **Correspondence**
- features.pt row i ↔ coordinates.npy row i ↔ same tile

## Customization

### Config Template
```yaml
# Input
wsi_path: /data/WSIs/WSI.svs
feature_path: /data/features/WSI.pt
coords_path: /data/coordinates/WSI_coordinates.npy

# Model
model_name: CLAM_MB_MIL
model_config_path: configs/CLAM_MB_MIL.yaml
checkpoint_path: logs/Best.pth

# Output
output_dir: logs/visualization/CLAM_MB_MIL/WSI

# Visualization
device: 0
cmap: jet           # viridis, hot, coolwarm
alpha: 0.4          # 0-1: transparency
vis_level: 1        # 0=highest res
patch_size: 512     # tile size
```

## Troubleshooting

### Row Mismatch Error
```python
# Check row counts
import torch, numpy as np
features = torch.load('features.pt')
coords = np.load('coordinates.npy')
print(f"Features: {features.shape[0]}")
print(f"Coords: {coords.shape[0]}")
# Should be equal!
```

### Wrong Coordinate Shape
```python
# Should be (N, 2)
coords = np.load('coordinates.npy')
assert coords.shape[1] == 2, "Need [x, y] pairs"
```

### CUDA Out of Memory
```yaml
# Reduce resolution
vis_level: 2        # Higher = lower res
thumbnail_size: [1024, 1024]
```

## Documentation
- `YOUR_FILE_FORMAT_GUIDE.md` - Your specific format
- `VISUALIZATION_README.md` - Full documentation
- `QUICKSTART.md` - 5-minute guide
- `CHANGES_SUMMARY.md` - What changed

## Quick Test
```bash
# 1. Validate
python vis_scripts/validate_input_files.py \
    --feature_path test.pt --coords_path test_coordinates.npy

# 2. Visualize
python complete_example.py \
    --wsi_dir /data/WSIs \
    --feature_dir /data/features \
    --coords_dir /data/coordinates \
    --wsi_name TEST \
    --model_name CLAM_MB_MIL \
    --model_config configs/CLAM_MB_MIL.yaml \
    --checkpoint logs/Best.pth

# 3. Check output
ls logs/visualization/CLAM_MB_MIL/TEST/
```

---
**Ready to visualize! 🎉**

# 1. Validate
$ python validate_input_files.py \
--feature_path /data_64T_3/Raja/Test/Extracted_tiles/Extracted_features/resnet50_1024/TCGA-BL-A13J-11A-01-TS1.02e153ec-4497-4d52-a6b1-6b80e50a426d.pt \
--coords_path /data_64T_3/Raja/Test/Extracted_tiles/Coordinates/TCGA-BL-A13J-11A-01-TS1.02e153ec-4497-4d52-a6b1-6b80e50a426d_coordinates.npy
 
# 2. Visualize
$ python visualize_heatmaps_single_wsi.py \
    --wsi_dir /data_64T_2/Dataset/TCGA_BLCA/images/Normal_TS \
    --feature_dir /data_64T_3/Raja/Test/Extracted_tiles/Extracted_features/resnet50_1024 \
    --coords_dir /data_64T_3/Raja/Test/Extracted_tiles/Coordinates \
    --wsi_name TCGA-BL-A13J-11A-01-TS1.02e153ec-4497-4d52-a6b1-6b80e50a426d \
    --model_name CLAM_MB_MIL \
    --model_config configs/CLAM_MB_MIL.yaml \
    --checkpoint logs/CLAM/CAMELYON16/CLAM_MB_MIL/resnet50_1024/time_2026-01-02-16-34_CAMELYON16_CLAM_MB_MIL_seed_41_resnet50_1024/fold_1/Best_EPOCH_195.pth 

# Single WSI
$ python visualize_heatmaps_single_wsi.py \
      --wsi_dir /data/WSIs \
      --feature_dir /data/features \
      --coords_dir /data/coordinates \
      --wsi_name TCGA-A1-A0SB \
      --model_name CLAM_MB_MIL \
      --model_config configs/CLAM_MB_MIL.yaml \
      --checkpoint logs/Best_EPOCH_57.pth
  
# Customize output
$ python visualize_heatmaps_single_wsi.py \
      --wsi_dir /data/WSIs \
      --feature_dir /data/features \
      --coords_dir /data/coordinates \
      --wsi_name TCGA-A1-A0SB \
      --model_name TDA_MIL \
      --model_config configs/TDA_MIL.yaml \
      --checkpoint logs/TDA_MIL/Best.pth \
      --output_dir logs/visualization/TDA_MIL/TCGA-A1-A0SB \
      --cmap viridis \
      --alpha 0.5 

Attention Score Analysis
========================
Analyze saved attention scores and extract top-k patches.

$ python analyze_attention_scores.py --attention_file WSI_NAME_attention.npy --top_k 100

Attention Heatmap Visualization for WSI Classification
======================================================
Given a trained MIL model, this script:
1. Loads WSI and pre-extracted features
2. Runs forward pass to get attention scores
3. Saves attention scores with coordinates
4. Generates heatmap overlay on original WSI

$ python generate_attention_heatmap.py --config vis_configs/attention_vis_config.yaml 

Validate Input Files for Attention Heatmap Visualization
========================================================
Check if your .pt features and .npy coordinates files are in the correct format.

$ python validate_input_files.py --feature_path features.pt --coords_path coords.npy

Complete Example - Your Exact File Format
==========================================
$ python visualize_heatmaps_single_wsi.py --wsi_dir /data/WSIs \
                          --feature_dir /data/features \
                          --coords_dir /data/coordinates \
                          --wsi_name TCGA-A1-A0SB \
                          --model_name CLAM_MB_MIL \
                          --checkpoint logs/Best_EPOCH_57.pth 

Batch Attention Heatmap Generation
===================================
Process multiple WSIs in batch mode.

$ python batch_generate_heatmaps.py --config vis_configs/batch_config.yaml 
