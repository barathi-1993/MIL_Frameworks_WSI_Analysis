import os
import sys
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import math
import openslide

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy.ndimage import gaussian_filter  # ✅ ADDED

import torch
import torch.nn.functional as F

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(grandparent_dir)

from utils.yaml_utils import read_yaml
from utils.model_utils import get_model_from_yaml

class AttentionHeatmapGenerator:
    """
    Generate attention heatmaps for whole slide images
    """
    
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary from YAML
        """
        self.config = config
        self.device = torch.device(f"cuda:{config.device}" if torch.cuda.is_available() else "cpu")
        
        # Paths
        self.wsi_path = config.wsi_path
        self.feature_path = config.feature_path
        self.coords_path = config.coords_path
        self.checkpoint_path = config.checkpoint_path
        self.output_dir = config.output_dir
        
        # Model config
        self.model_config_path = config.model_config_path
        self.model_name = config.model_name
        
        # Visualization params
        self.cmap = config.get('cmap', 'coolwarm')
        self.alpha = config.get('alpha', 0.4)
        self.vis_level = config.get('vis_level', 1)
        self.thumbnail_size = config.get('thumbnail_size', None)
        self.sigma = config.get('sigma', 0.6)  # ✅ ADDED: Gaussian smoothing parameter
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # WSI name from path
        self.wsi_name = Path(self.wsi_path).stem
        
        print(f"[INFO] Initializing AttentionHeatmapGenerator")
        print(f"  WSI: {self.wsi_name}")
        print(f"  Model: {self.model_name}")
        print(f"  Device: {self.device}")
        print(f"  Output: {self.output_dir}")
        
    def load_model(self):
        # 1. Get the project root directory (grandparent of this script)
        # /home/.../1.WSI_Classification/vis_scripts/ -> /home/.../1.WSI_Classification/
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        
        # 2. Construct the absolute path to the config
        # This forces Python to look in /home/.../1.WSI_Classification/configs/CLAM_MB_MIL.yaml
        config_path = os.path.join(project_root, self.model_config_path)        
        print(f"Loading config from: {config_path}") # Debug print to confirm
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at: {config_path}")

        model_args = read_yaml(config_path)
        
        # Get model architecture
        if self.model_name == 'CLAM_MB_MIL':
            from modules.CLAM_MB_MIL.clam_mb_mil import CLAM_MB_MIL
            from utils.process_utils import get_act
            
            self.model = CLAM_MB_MIL(
                gate=model_args.Model.gate,
                size_arg=model_args.Model.size_arg,
                dropout=model_args.Model.dropout,
                k_sample=model_args.Model.k_sample,
                num_classes=model_args.General.num_classes,
                instance_loss_fn=None,  # Not needed for inference
                subtyping=model_args.Model.subtyping,
                embed_dim=model_args.Model.in_dim,
                act=get_act(model_args.Model.act),
                instance_eval=False  # Disable instance eval for inference
            )
        elif self.model_name == 'AC_MIL':
            # Adjust the import path based on where you saved ac_mil.py
            from modules.AC_MIL.ac_mil import AC_MIL 
            
            self.model = AC_MIL(
                in_dim=model_args.Model.in_dim,
                hidden_dim=model_args.Model.hidden_dim,
                num_classes=model_args.General.num_classes,
                n_token=model_args.Model.n_token,
                n_masked_patch=model_args.Model.n_masked_patch,
                mask_drop=model_args.Model.mask_prob # Note: YAML uses mask_prob, Class uses mask_drop
            )
            
        elif self.model_name == 'DeepAttn_MIL':
            # Adjust the import path based on where you saved deep_attn_mil.py
            from modules.DeepAttn_MIL.deep_attn_mil import DeepAttnMIL_Surv
                
            self.model = DeepAttnMIL_Surv(
                in_dim=model_args.Model.in_dim,
                embedding_dim=model_args.Model.embedding_dim,
                attention_dim=model_args.Model.attention_dim,
                fc_dim=model_args.Model.fc_dim,
                num_classes=model_args.General.num_classes,
                dropout=model_args.Model.dropout,
                cluster_num=model_args.Model.cluster_num,
                act=model_args.Model.act
            )
            
        elif self.model_name == 'AB_MIL':
            from modules.AB_MIL.ab_mil import AB_MIL
            from utils.process_utils import get_act
            
            self.model = AB_MIL(
                L=model_args.Model.L,
                D=model_args.Model.D,
                num_classes=model_args.General.num_classes,
                dropout=model_args.Model.dropout,
                act=get_act(model_args.Model.act),
                in_dim=model_args.Model.in_dim
            )
            
        elif self.model_name in ['MAMBA_MIL', 'BiMAMBA_MIL', 'SRMAMBA_MIL']:
            from modules.MAMBA_MIL.MambaMIL import MambaMIL
            
            mamba_type = {'MAMBA_MIL': 'Mamba', 'BiMAMBA_MIL': 'BiMamba', 'SRMAMBA_MIL': 'SRMamba'}
            self.model = MambaMIL(
                in_dim=model_args.Model.in_dim,
                n_classes=model_args.General.num_classes,
                dropout=model_args.Model.dropout,
                type=mamba_type[self.model_name]
            )
            
        elif self.model_name == 'TDA_MIL':
            from modules.TDA_MIL.tda_mil import TDA_MIL
            
            self.model = TDA_MIL(
                in_dim=model_args.Model.in_dim,
                embed_dim=model_args.Model.embed_dim,
                num_classes=model_args.General.num_classes,
                num_layers=model_args.Model.num_layers,
                num_heads=model_args.Model.num_heads,
                mlp_ratio=model_args.Model.mlp_ratio,
                dropout=model_args.Model.dropout,
                attn_dropout=model_args.Model.attn_dropout,
                td_mlp_ratio=model_args.Model.td_mlp_ratio,
                clamp_min=model_args.Model.clamp_min,
                clamp_max=model_args.Model.clamp_max,
                force_cls_score=model_args.Model.force_cls_score,
                share_weights_step12=model_args.Model.share_weights_step12,
                max_seq_len=model_args.Model.max_seq_len
            ) 

        else:
            raise ValueError(f"Model {self.model_name} not supported")
        
        # Load checkpoint
        model_path = os.path.join(project_root, self.checkpoint_path)
        state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        print(f"  ✓ Model loaded from: {self.checkpoint_path}")
        
    def load_features_and_coords(self):
        """Load pre-extracted features and coordinates"""
        print(f"\n[STEP 2] Loading features and coordinates...")
        
        # Load features (.pt file)
        features = torch.load(self.feature_path)
        if isinstance(features, dict):
            if 'feats' in features:
                features = features['feats']
            elif 'features' in features:
                features = features['features']
        
        if len(features.shape) == 3:
            features = features.squeeze(0)
        
        self.features = features  # (N, D)
        print(f"  ✓ Features loaded: {self.features.shape}")
        
        # Load coordinates (.npy file)
        # Format: [[x1, y1], [x2, y2], ...] where row i corresponds to feature row i
        self.coords = np.load(self.coords_path)  # (N, 2) - [x, y] coordinates
        
        # Ensure coords is 2D array with shape (N, 2)
        if len(self.coords.shape) == 1:
            raise ValueError(f"Coordinates should be (N, 2) array, got shape {self.coords.shape}")
        
        # ✅ ADDED: Sort features and coords together for alignment
        print(f"  ✓ Sorting features and coordinates for alignment...")
        order = np.lexsort((self.coords[:, 0], self.coords[:, 1]))
        self.coords = self.coords[order]
        self.features = self.features[order]
        
        print(f"  ✓ Coordinates loaded: {self.coords.shape}")
        print(f"  ✓ Coordinate format: [x, y] pairs")
        print(f"  ✓ X range: [{self.coords[:, 0].min()}, {self.coords[:, 0].max()}]")
        print(f"  ✓ Y range: [{self.coords[:, 1].min()}, {self.coords[:, 1].max()}]")
        
        # Verify row-by-row correspondence
        assert self.features.shape[0] == self.coords.shape[0], \
            f"Mismatch: {self.features.shape[0]} features vs {self.coords.shape[0]} coordinates"
        
        assert self.coords.shape[1] == 2, \
            f"Coordinates should have 2 columns [x, y], got {self.coords.shape[1]}"
        
        self.num_patches = self.features.shape[0]
        print(f"  ✓ Total patches: {self.num_patches}")
        print(f"  ✓ Row-by-row correspondence verified")
        
    def extract_attention_scores(self):
        """Run forward pass and extract attention scores"""
        print(f"\n[STEP 3] Extracting attention scores...")
        
        # Move features to device
        features = self.features.to(self.device)
        features = features.unsqueeze(0)  # (1, N, D)
        
        expected_dim = self.model.attention_net[0].in_features if hasattr(self.model, 'attention_net') else self.config.get('in_dim', 1024)
        actual_dim = features.shape[2]
        
        if actual_dim != expected_dim:
            # Try to grab the first layer dimension dynamically if possible to be precise
            try:
                # For CLAM, the first layer is usually in attention_net
                model_dim = self.model.attention_net[0].in_features
            except:
                model_dim = "Unknown (likely 1024)"

            print(f"\n❌ DIMENSION MISMATCH ERROR:")
            print(f"   Your loaded features have dimension: {actual_dim}")
            print(f"   Your model expects dimension:        {model_dim}")
            print(f"   Please check your 'feature_path' in the config or update 'in_dim' in the model yaml.\n")
            sys.exit(1)

        with torch.no_grad():
            # Forward pass with attention return
            forward_return = self.model(features, return_WSI_attn=True, return_WSI_feature=True)
            
            logits = forward_return['logits']  # (1, num_classes)
            attention = forward_return['WSI_attn']  # Model-dependent shape
            
            # Get prediction
            probs = F.softmax(logits, dim=1)
            pred_class = torch.argmax(probs, dim=1).item()
            pred_prob = probs[0, pred_class].item()
            
            print(f"  ✓ Prediction: Class {pred_class} (confidence: {pred_prob:.4f})")
            
            # Process attention based on model type
            if self.model_name == 'CLAM_MB_MIL':
                # CLAM returns attention for predicted class: (1, N) or (N,)
                if len(attention.shape) == 3:
                    attention = attention.squeeze(0) 
                if len(attention.shape) == 2:
                    attention = attention.squeeze(0)  

            elif self.model_name == 'AC_MIL':
                # AC_MIL returns raw averaged logits in 'WSI_attn'
                # Shape is typically (N, 1) or (1, N, 1) depending on batching
                if len(attention.shape) == 3:
                    attention = attention.squeeze(0)  # (1, N, 1) -> (N, 1)
                
                if attention.shape[-1] == 1:
                    attention = attention.squeeze(-1) # (N, 1) -> (N,)
                
                # AC_MIL uses softmax in the forward pass for aggregation: 
                # feat_bag = self.bag_attention(v, attns.softmax(dim=-1)...)
                # So we must apply softmax here to visualize the actual weights
                attention = F.softmax(attention, dim=0)

            elif self.model_name == 'DeepAttn_MIL':
                # DeepAttn_MIL returns normalized attention in 'attention' key
                # Shape is typically [1, N]
                if 'attention' in forward_return:
                    attention = forward_return['attention']
                elif 'WSI_attn' in forward_return:
                    attention = forward_return['WSI_attn']
                else:
                    # Fallback if using a different return key in a specific version
                    attention = attention 

                if len(attention.shape) == 2:
                    attention = attention.squeeze(0)  # (1, N) -> (N,)
                
                # Attention is already softmaxed in the model's forward pass
                # A = F.softmax(A, dim=1) inside deep_attn_mil.py
                                   
            elif self.model_name == 'AB_MIL':
                # AB_MIL returns raw attention before softmax: (N, 1)
                if len(attention.shape) == 3:
                    attention = attention.squeeze(0)  # (1, N, 1) -> (N, 1)
                if attention.shape[-1] == 1:
                    attention = attention.squeeze(-1)  # (N, 1) -> (N,)
                # Apply softmax to get attention weights
                attention = F.softmax(attention, dim=0) 
                
            elif self.model_name in ['MAMBA_MIL', 'BiMAMBA_MIL', 'SRMAMBA_MIL']:
                # Mamba returns raw attention: (1, N, 1) or similar
                if len(attention.shape) == 3:
                    attention = attention.squeeze(0)  # (1, N, 1) -> (N, 1)
                if attention.shape[-1] == 1:
                    attention = attention.squeeze(-1)  # (N, 1) -> (N,)
                # Apply softmax
                attention = attention.transpose(0, 1) if len(attention.shape) == 2 else attention
                attention = F.softmax(attention, dim=-1)
                if len(attention.shape) == 2:
                    attention = attention.squeeze(0) 
                
            elif self.model_name == 'TDA_MIL':
                # TDA_MIL returns selection scores (already in [0, 1]): (N,)
                if len(attention.shape) > 1:
                    attention = attention.squeeze()             
            else:
                raise ValueError(f"Attention extraction not implemented for {self.model_name}")

            # Convert to numpy first
            attention_scores = attention.cpu().numpy()
            
            # ✅ ADDED: Convert to percentile ranks for better visualization
            print(f"  ✓ Converting attention to percentile ranks...")
            attention_scores = self._rank_percentiles(attention_scores)
                
            # Ensure correct shape
            if attention_scores.shape[0] != self.num_patches:
                print(f"  ! Warning: Attention shape {attention_scores.shape} != {self.num_patches}")
                # Handle potential mismatch (e.g., TDA_MIL sampling)
                if attention_scores.shape[0] < self.num_patches:
                    # Pad with zeros
                    padded = np.zeros(self.num_patches)
                    padded[:attention_scores.shape[0]] = attention_scores
                    attention_scores = padded
                else:
                    attention_scores = attention_scores[:self.num_patches]
            
            self.attention_scores = attention_scores
            self.pred_class = pred_class
            self.pred_prob = pred_prob
            
            print(f"  ✓ Attention scores extracted: {self.attention_scores.shape}")
            print(f"  ✓ Attention range: [{self.attention_scores.min():.6f}, {self.attention_scores.max():.6f}]")
            print(f"  ✓ Attention mean: {self.attention_scores.mean():.6f}")
    
    # ✅ ADDED: Percentile ranking function
    def _rank_percentiles(self, x: np.ndarray) -> np.ndarray:
        """
        Fast percentile ranks in [0,1], O(N log N).
        This provides better visual contrast than linear min-max normalization.
        """
        x = np.asarray(x).reshape(-1)
        order = np.argsort(x, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float32)
        ranks[order] = np.arange(len(x), dtype=np.float32)
        if len(x) > 1:
            ranks /= (len(x) - 1)
        return ranks
    
    def save_attention_with_coords(self):
        """Save attention scores with tile coordinates"""
        print(f"\n[STEP 4] Saving attention scores...")
        
        # Create array: [x, y, attention]
        attention_with_coords = np.column_stack([
            self.coords[:, 0],  # x coordinate
            self.coords[:, 1],  # y coordinate
            self.attention_scores
        ])
        
        # Save as .npy
        output_path = os.path.join(self.output_dir, f"{self.wsi_name}_attention.npy")
        np.save(output_path, attention_with_coords)
        print(f"  ✓ Saved attention with coords: {output_path}")
        print(f"    Shape: {attention_with_coords.shape} [x, y, attention]")
        
        # Also save as readable .txt
        txt_path = os.path.join(self.output_dir, f"{self.wsi_name}_attention.txt")
        np.savetxt(txt_path, attention_with_coords, 
                   header="x_coord y_coord attention_score",
                   fmt='%d %d %.8f')
        print(f"  ✓ Saved readable format: {txt_path}")
        
        return output_path 

    def get_scale_factor(self, slide):              
        # Check for the scale factor
        down_samples = list(slide.level_downsamples)
        rounded_down_samples = {round(v) for v in down_samples}  
     
        if 32 in rounded_down_samples:
            scale_factor = 32
        elif 16 in rounded_down_samples:
            scale_factor = 16
        elif 4 in rounded_down_samples:
            scale_factor = 4
        elif 1 in rounded_down_samples:    
            scale_factor = 1
        else: 
            # Default fallback if specific levels aren't found
            scale_factor = 32
            print(f"  ! Warning: Standard downsamples not found. Defaulting to {scale_factor}")
            
        return scale_factor  

    def get_downsampled_image(self, slide, scale_factor): 
        large_w, large_h = slide.dimensions
        new_w = math.floor(large_w/scale_factor)
        new_h = math.floor(large_h/scale_factor)
        
        # Default to thumbnail
        img = slide.get_thumbnail((new_w, new_h)).convert("RGB")
        
        try:  
            level = slide.get_best_level_for_downsample(scale_factor) 
            whole_slide_image = slide.read_region((0, 0), level, slide.level_dimensions[level])
            whole_slide_image = whole_slide_image.convert("RGB")
            img = whole_slide_image.resize((new_w, new_h), Image.BILINEAR) 
            return img, new_w, new_h
        except Exception as e:
            print(f"  ! Warning during downsampling: {e}. Using thumbnail.")
            return img, new_w, new_h

    def generate_heatmap(self):
        """Generate attention heatmap overlay on WSI"""
        print(f"\n[STEP 5] Generating heatmap...")
              
        # Load WSI
        slide = openslide.OpenSlide(self.wsi_path)
        
        # ✅ FIX: Get tissue bounding box from coordinates
        # x_coords = self.coords[:, 0]
        # y_coords = self.coords[:, 1]
        
        # min_x = int(x_coords.min())
        # min_y = int(y_coords.min())
        # max_x = int(x_coords.max())
        # max_y = int(y_coords.max())
        
        # print(f"  ✓ Tissue bounding box:")
        # print(f"    X: [{min_x}, {max_x}]")
        # print(f"    Y: [{min_y}, {max_y}]")

        # We explicitly set the origin to (0,0) to capture the whole slide
        min_x = 0
        min_y = 0
        
        # We use the full slide dimensions
        tissue_width, tissue_height = slide.dimensions
        
        # We don't use max_x/max_y for canvas size anymore, but we can keep 
        # coords for the mapping loop later.
        
        print(f"  ✓ Using Full WSI Dimensions:")
        print(f"    Size: {tissue_width} x {tissue_height}")
        print(f"    Origin: ({min_x}, {min_y})")

        # Get patch size (Logic kept same, just needed for the loop later)
        if hasattr(self.config, 'patch_size') and self.config.patch_size is not None and self.config.patch_size > 0:
            patch_size = self.config.patch_size
        else:
            unique_x = np.unique(self.coords[:, 0])
            if len(unique_x) >= 2:
                patch_size = int(unique_x[1] - unique_x[0])
            else:
                patch_size = 256
        
        # Get patch size
        if hasattr(self.config, 'patch_size') and self.config.patch_size is not None and self.config.patch_size > 0:
            patch_size = self.config.patch_size
        else:
            unique_x = np.unique(self.coords[:, 0])
            if len(unique_x) >= 2:
                patch_size = int(unique_x[1] - unique_x[0])
            else:
                patch_size = 256
        
        print(f"  ✓ Patch size: {patch_size}px")
        
        # # ✅ FIX: Calculate tissue region size (not entire WSI)
        # tissue_width = max_x - min_x + patch_size
        # tissue_height = max_y - min_y + patch_size
        
        print(f"  ✓ Tissue region size: {tissue_width} x {tissue_height}")
        
        # ✅ FIX: Determine downsample based on tissue size
        # Target: canvas around 2000-4000 pixels on longest side
        target_max_dim = 3000
        downsample_x = tissue_width / target_max_dim
        downsample_y = tissue_height / target_max_dim
        downsample = max(downsample_x, downsample_y, 1.0)
        
        # Round to nearest power of 2 or common downsample
        if downsample <= 2:
            downsample = 1
        elif downsample <= 8:
            downsample = 4
        elif downsample <= 24:
            downsample = 16
        else:
            downsample = 32
        
        print(f"  ✓ Using downsample: {downsample}x")
        
        # ✅ FIX: Read ONLY the tissue region
        level = slide.get_best_level_for_downsample(downsample)
        level_downsample = slide.level_downsamples[level]
        
        # Calculate read size at chosen level
        read_w = int(tissue_width / level_downsample)
        read_h = int(tissue_height / level_downsample)
        
        print(f"  ✓ Reading region at level {level} (downsample={level_downsample:.2f})")
        print(f"    Read size: {read_w} x {read_h}")
        print(f"    Region start: ({min_x}, {min_y})")
        
        # Read tissue region only
        wsi_region = slide.read_region((min_x, min_y), level, (read_w, read_h))
        wsi_region = wsi_region.convert("RGB")
        
        # Resize to final canvas size
        canvas_w = int(tissue_width / downsample)
        canvas_h = int(tissue_height / downsample)
        
        wsi_img = wsi_region.resize((canvas_w, canvas_h), Image.Resampling.LANCZOS)
        wsi_img = np.array(wsi_img)
        
        print(f"  ✓ Final canvas size: {canvas_w} x {canvas_h}")
        print(f"  ✓ WSI image loaded: {wsi_img.shape}")
        
        # Create heatmap canvas (matches tissue region)
        heatmap = np.zeros((canvas_h, canvas_w))
        counts = np.zeros((canvas_h, canvas_w))
        
        # ✅ FIX: Map coordinates relative to tissue origin
        print(f"  ✓ Mapping {self.num_patches} patches to heatmap...")
        for idx in tqdm(range(self.num_patches), desc="  Mapping patches"):
            x, y = self.coords[idx]
            attn = self.attention_scores[idx]
            
            # Convert to tissue-relative coordinates
            x_rel = x - min_x
            y_rel = y - min_y
            
            # Convert to canvas coordinates
            x_canvas = int(x_rel / downsample)
            y_canvas = int(y_rel / downsample)
            patch_size_canvas = int(patch_size / downsample)
            
            # Skip if out of bounds
            if (x_canvas < 0 or y_canvas < 0 or 
                x_canvas + patch_size_canvas > canvas_w or 
                y_canvas + patch_size_canvas > canvas_h):
                continue
            
            # Fill patch region
            heatmap[y_canvas:y_canvas+patch_size_canvas, 
                    x_canvas:x_canvas+patch_size_canvas] += attn
            counts[y_canvas:y_canvas+patch_size_canvas, 
                   x_canvas:x_canvas+patch_size_canvas] += 1
        
        # Average overlapping regions
        heatmap = np.divide(heatmap, counts, where=counts > 0)
        
        # Gaussian smoothing
        print(f"  ✓ Applying Gaussian smoothing (sigma={self.sigma})...")
        smooth_heatmap = gaussian_filter(heatmap, sigma=self.sigma)
        smooth_counts = gaussian_filter(counts, sigma=self.sigma)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            heatmap = smooth_heatmap / smooth_counts
            # ✅ CHANGE: Increase the threshold to cut off the "smoothing fuzz"
            # Any tiny value created by smoothing is forced back to 0.0
            heatmap[smooth_counts < 0.1] = 0.0
            heatmap[heatmap < 0.001] = 0.0 
        
        # Normalize to [0, 1]
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
            
        # ✅ CHANGE: Force clean zero background again after normalization
        # This ensures the background is absolute 0.0 before coloring
        heatmap[heatmap < 0.01] = 0.0

        print(f"  ✓ Heatmap generated: {heatmap.shape}")
        print(f"    Coverage: {(counts > 0).sum() / counts.size * 100:.2f}%")
        
        # Apply colormap
        # cmap_fn = plt.get_cmap(self.cmap)
        # heatmap_colored = cmap_fn(heatmap)[:, :, :3]
        # heatmap_colored = (heatmap_colored * 255).astype(np.uint8)

        # Apply colormap
        cmap_fn = plt.get_cmap(self.cmap)
        heatmap_colored = cmap_fn(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # ✅ CHANGE: Hard overwrite background pixels to black
        # Using a slightly higher threshold (0.01) ensures no dark blue artifacts remain
        heatmap_colored[heatmap < 0.01] = [0, 0, 0]
        
        # Create overlay
        overlay = cv2.addWeighted(wsi_img, 1 - self.alpha, heatmap_colored, self.alpha, 0)
        
        # Save outputs
        output_path = os.path.join(self.output_dir, f"{self.wsi_name}_heatmap.png")
        Image.fromarray(overlay).save(output_path)
        print(f"  ✓ Saved heatmap overlay: {output_path}")
        
        heatmap_only_path = os.path.join(self.output_dir, f"{self.wsi_name}_heatmap_only.png")
        Image.fromarray(heatmap_colored).save(heatmap_only_path)
        print(f"  ✓ Saved heatmap only: {heatmap_only_path}")
        
        orig_path = os.path.join(self.output_dir, f"{self.wsi_name}_original.png")
        Image.fromarray(wsi_img).save(orig_path)
        print(f"  ✓ Saved original: {orig_path}")
        
        self._create_summary_figure(wsi_img, heatmap, overlay)
        
        slide.close()
        return output_path
    
    def _generate_heatmap_blank_canvas(self):
        """Generate heatmap on blank canvas when OpenSlide unavailable"""
        print("  ✓ Generating heatmap on blank canvas...")
        
        # Get bounding box
        x_coords = self.coords[:, 0]
        y_coords = self.coords[:, 1]
        
        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()
        
        patch_size = self.config.get('patch_size', 256)
        
        # Create canvas
        canvas_width = int((x_max - x_min + patch_size) / 4)  # Downsample 4x
        canvas_height = int((y_max - y_min + patch_size) / 4)
        
        heatmap = np.zeros((canvas_height, canvas_width))
        
        # Map patches
        for idx in range(self.num_patches):
            x, y = self.coords[idx]
            attn = self.attention_scores[idx]
            
            x_canvas = int((x - x_min) / 4)
            y_canvas = int((y - y_min) / 4)
            patch_size_canvas = int(patch_size / 4)
            
            if (x_canvas >= 0 and y_canvas >= 0 and 
                x_canvas + patch_size_canvas <= canvas_width and 
                y_canvas + patch_size_canvas <= canvas_height):
                heatmap[y_canvas:y_canvas+patch_size_canvas, 
                       x_canvas:x_canvas+patch_size_canvas] = attn
        
        # Normalize
        if heatmap.max() > 0:
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
        
        # Apply colormap
        cmap_fn = cm.get_cmap(self.cmap)
        heatmap_colored = cmap_fn(heatmap)[:, :, :3]
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Save
        output_path = os.path.join(self.output_dir, f"{self.wsi_name}_heatmap.png")
        Image.fromarray(heatmap_colored).save(output_path)
        print(f"  ✓ Saved heatmap: {output_path}")
        
        return output_path
    
    def _create_summary_figure(self, original, heatmap, overlay):
        """Create summary figure with all three views"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 1. Original
        axes[0].imshow(original)
        axes[0].set_title('Original WSI', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # 2. Attention Heatmap (Scalar Plot with Colorbar)
        axes[1].set_facecolor('black')
        
        # ✅ CHANGE: Robust masking strategy
        # 1. Copy colormap and set "bad" (masked) values to black
        my_cmap = plt.get_cmap(self.cmap).copy()
        my_cmap.set_bad(color='black')
        
        # 2. Mask out the background (values close to 0)
        # Anything masked will be colored using 'set_bad' (black)
        heatmap_masked = np.ma.masked_less(heatmap, 0.01)
        
        # 3. Plot with interpolation='nearest'
        # 'nearest' prevents Matplotlib from blurring the edges (which creates blue halos)
        im = axes[1].imshow(heatmap_masked, cmap=my_cmap, vmin=0, vmax=1, interpolation='nearest')
        axes[1].set_title(f'Attention Heatmap ({self.model_name})', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        # Add colorbar to the heatmap axis
        cbar = plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
        cbar.set_label('Attention Percentile', rotation=270, labelpad=15)  # ✅ UPDATED label
        
        # 3. Overlay
        axes[2].imshow(overlay)
        axes[2].set_title(f'Overlay (α={self.alpha})', fontsize=14, fontweight='bold')
        axes[2].axis('off')
        
        plt.suptitle(f'{self.wsi_name} - Prediction: Class {self.pred_class} ({self.pred_prob:.2%})',
                     fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        summary_path = os.path.join(self.output_dir, f"{self.wsi_name}_summary.png")
        plt.savefig(summary_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved summary figure: {summary_path}")
    
    def run(self):
        """Run complete pipeline"""
        print("\n" + "="*70)
        print(f"ATTENTION HEATMAP GENERATION")
        print("="*70)
        
        self.load_model()
        self.load_features_and_coords()
        self.extract_attention_scores()
        self.save_attention_with_coords()
        self.generate_heatmap()
        
        print("\n" + "="*70)
        print("✓ COMPLETE!")
        print("="*70)
        print(f"\nOutput files saved to: {self.output_dir}")
        print(f"  - {self.wsi_name}_attention.npy (attention scores with coords)")
        print(f"  - {self.wsi_name}_attention.txt (readable format)")
        print(f"  - {self.wsi_name}_heatmap.png (heatmap overlay)")
        print(f"  - {self.wsi_name}_heatmap_only.png (heatmap without overlay)")
        print(f"  - {self.wsi_name}_original.png (original thumbnail)")
        print(f"  - {self.wsi_name}_summary.png (summary figure)")
        print()

def main():
    parser = argparse.ArgumentParser(description='Generate attention heatmaps for WSI')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to visualization config YAML')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert to namespace for dot notation
    class Config:
        def __init__(self, d):
            for key, value in d.items():
                if isinstance(value, dict):
                    setattr(self, key, Config(value))
                else:
                    setattr(self, key, value)
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    config = Config(config_dict)
    
    # Run visualization
    generator = AttentionHeatmapGenerator(config)
    generator.run()

if __name__ == '__main__':
    main()