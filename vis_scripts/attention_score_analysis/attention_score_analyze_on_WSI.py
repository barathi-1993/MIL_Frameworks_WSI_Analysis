# # Basic usage with input file
# python analyze_attention_score_on_WSI.py --input attention_scores.txt

# # Specify output directory
# python analyze_attention_score_on_WSI.py --input attention_scores.txt --output /path/to/output

# # Specify top-k patches
# python analyze_attention_score_on_WSI.py --input attention_scores.txt --top_k 50

# # Specify all the options
# python analyze_attention_score_on_WSI.py --input attention_scores.txt  --top_k 50 --output /path/to/output

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from pathlib import Path

class AttentionAnalyzer:
    """
    Analyze attention scores from WSI
    """
    
    def __init__(self, attention_file, output_dir=None):
        # Set input file path
        self.attention_file = Path(attention_file)
        
        # Set output directory
        if output_dir is None:
            # Default to a subfolder in the input file's parent directory
            self.output_dir = self.attention_file.parent / f"{self.attention_file.stem}_analysis"
        else:
            self.output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # WSI name (use filename without extension)
        self.wsi_name = self.attention_file.stem
        
        # Load attention data
        self.data = self.load_attention_data(self.attention_file)
        self.x_coords = self.data[:, 0].astype(int)
        self.y_coords = self.data[:, 1].astype(int)
        self.attention = self.data[:, 2]
        
        print(f"Loaded attention data for: {self.wsi_name}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Total patches: {len(self.attention)}")
        print(f"  Attention range: [{self.attention.min():.6f}, {self.attention.max():.6f}]")
        print(f"  Attention mean: {self.attention.mean():.6f} ± {self.attention.std():.6f}")
        print()

    def load_attention_data(self, file_path):
        """
        Load attention data from different file formats
        
        Args:
            file_path (Path or str): Path to attention data file
        
        Returns:
            np.ndarray: Attention data with columns [x_coord, y_coord, attention_score]
        """
        file_path = Path(file_path)
        file_ext = file_path.suffix.lower()
        
        if file_ext == '.npy':
            # NumPy binary file
            return np.load(file_path)
        elif file_ext in ['.txt', '.csv']:
            # Text or CSV file
            try:
                # Try reading as space-separated values, ignoring comments
                df = pd.read_csv(file_path, 
                                 delim_whitespace=True,  # Space-separated
                                 comment='#',            # Ignore comment lines
                                 header=None,            # No header
                                 names=['x_coord', 'y_coord', 'attention_score'])
                
                return df[['x_coord', 'y_coord', 'attention_score']].values
            except Exception as e:
                print(f"Error reading text file: {e}")
                # Fallback to pandas read_csv with more flexible parsing
                df = pd.read_csv(file_path, 
                                 comment='#',  # Ignore comment lines
                                 header=None)
                
                # Assume last column is attention score, first two are coordinates
                return df.iloc[:, :3].values
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def get_statistics(self):
        """Calculate attention score statistics"""
        stats = {
            'mean': self.attention.mean(),
            'std': self.attention.std(),
            'min': self.attention.min(),
            'max': self.attention.max(),
            'median': np.median(self.attention),
            'q25': np.percentile(self.attention, 25),
            'q75': np.percentile(self.attention, 75),
            'q95': np.percentile(self.attention, 95),
            'q99': np.percentile(self.attention, 99)
        }
        
        return stats
    
    def plot_distribution(self, save=True):
        """Plot attention score distribution with save option"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Histogram
        axes[0].hist(self.attention, bins=50, edgecolor='black', alpha=0.7)
        axes[0].axvline(self.attention.mean(), color='red', linestyle='--', 
                       label=f'Mean: {self.attention.mean():.6f}')
        axes[0].axvline(np.median(self.attention), color='green', linestyle='--',
                       label=f'Median: {np.median(self.attention):.6f}')
        axes[0].set_xlabel('Attention Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Attention Score Distribution')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Box plot
        axes[1].boxplot(self.attention, vert=True)
        axes[1].set_ylabel('Attention Score')
        axes[1].set_title('Attention Score Box Plot')
        axes[1].grid(alpha=0.3)
        
        plt.suptitle(f'{self.wsi_name} - Attention Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save:
            output_path = self.output_dir / f"{self.wsi_name}_attention_distribution.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved distribution plot: {output_path}")
        
        plt.close()
    
    def plot_spatial_distribution(self, save=True):
        """Plot spatial distribution of attention scores"""
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        
        scatter = ax.scatter(self.x_coords, self.y_coords, 
                            c=self.attention, cmap='coolwarm', s=10, alpha=0.6)
        
        plt.colorbar(scatter, ax=ax, label='Attention Score')
        ax.set_xlabel('X Coordinate (pixels)')
        ax.set_ylabel('Y Coordinate (pixels)')
        ax.set_title(f'{self.wsi_name} - Spatial Attention Distribution')
        ax.invert_yaxis()  # Invert Y axis to match image coordinates
        ax.set_aspect('equal')
        
        if save:
            output_path = self.output_dir / f"{self.wsi_name}_spatial_attention.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved spatial plot: {output_path}")
        
        plt.close()
    
    def get_top_k_patches(self, k=100):
        """Get top-k patches by attention score"""
        top_k_idx = np.argsort(self.attention)[-k:][::-1]  # Descending order
        
        top_k_data = {
            'rank': np.arange(1, k+1),
            'patch_idx': top_k_idx,
            'x_coord': self.x_coords[top_k_idx],
            'y_coord': self.y_coords[top_k_idx],
            'attention': self.attention[top_k_idx]
        }
        
        return pd.DataFrame(top_k_data)
    
    def get_bottom_k_patches(self, k=100):
        """Get bottom-k patches by attention score"""
        bottom_k_idx = np.argsort(self.attention)[:k]
        
        bottom_k_data = {
            'rank': np.arange(1, k+1),
            'patch_idx': bottom_k_idx,
            'x_coord': self.x_coords[bottom_k_idx],
            'y_coord': self.y_coords[bottom_k_idx],
            'attention': self.attention[bottom_k_idx]
        }
        
        return pd.DataFrame(bottom_k_data)
    
    def get_patches_above_threshold(self, threshold):
        """Get patches with attention above threshold"""
        high_attn_idx = np.where(self.attention >= threshold)[0]
        
        data = {
            'patch_idx': high_attn_idx,
            'x_coord': self.x_coords[high_attn_idx],
            'y_coord': self.y_coords[high_attn_idx],
            'attention': self.attention[high_attn_idx]
        }
        
        return pd.DataFrame(data)
    
    def plot_top_k_heatmap(self, k=100, save=True):
        """Plot heatmap of top-k patches"""
        top_k_idx = np.argsort(self.attention)[-k:]
        
        # Create grid
        x_min, x_max = self.x_coords.min(), self.x_coords.max()
        y_min, y_max = self.y_coords.min(), self.y_coords.max()
        
        # Estimate patch size
        unique_x = sorted(np.unique(self.x_coords))
        if len(unique_x) > 1:
            patch_size = unique_x[1] - unique_x[0]
        else:
            patch_size = 512
        
        # Create binary mask
        grid_width = int((x_max - x_min) / patch_size) + 1
        grid_height = int((y_max - y_min) / patch_size) + 1
        
        mask = np.zeros((grid_height, grid_width))
        
        for idx in top_k_idx:
            x_grid = int((self.x_coords[idx] - x_min) / patch_size)
            y_grid = int((self.y_coords[idx] - y_min) / patch_size)
            if 0 <= x_grid < grid_width and 0 <= y_grid < grid_height:
                mask[y_grid, x_grid] = 1
        
        # Plot
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(mask, cmap='Reds', interpolation='nearest')
        ax.set_xlabel('Grid X')
        ax.set_ylabel('Grid Y')
        ax.set_title(f'{self.wsi_name} - Top {k} Patches')
        
        if save:
            output_path = self.output_dir / f"{self.wsi_name}_top{k}_mask.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved top-k mask: {output_path}")
        
        plt.close()
    
    def save_top_k_patches(self, k=100):
        """Save top-k patch information"""
        top_k_df = self.get_top_k_patches(k)
        
        output_path = self.output_dir / f"{self.wsi_name}_top{k}_patches.csv"
        top_k_df.to_csv(output_path, index=False)
        print(f"✓ Saved top-{k} patches: {output_path}")
        
        return top_k_df
    
    # def generate_report(self, top_k=100):
    #     """Generate comprehensive analysis report"""
    #     print("\n" + "="*70)
    #     print(f"ATTENTION SCORE ANALYSIS REPORT - {self.wsi_name}")
    #     print("="*70)
        
    #     # Statistics
    #     stats = self.get_statistics()
    #     print("\n[1] Statistics:")
    #     print(f"  Total Patches: {len(self.attention)}")
    #     print(f"  Mean:          {stats['mean']:.6f}")
    #     print(f"  Std Dev:       {stats['std']:.6f}")
    #     print(f"  Min:           {stats['min']:.6f}")
    #     print(f"  Max:           {stats['max']:.6f}")
    #     print(f"  Median:        {stats['median']:.6f}")
    #     print(f"  Q25:           {stats['q25']:.6f}")
    #     print(f"  Q75:           {stats['q75']:.6f}")
    #     print(f"  Q95:           {stats['q95']:.6f}")
    #     print(f"  Q99:           {stats['q99']:.6f}")
        
    #     # Top patches
    #     print(f"\n[2] Top-{top_k} Patches:")
    #     top_k_df = self.get_top_k_patches(top_k)
    #     print(f"  Attention range: [{top_k_df['attention'].min():.6f}, {top_k_df['attention'].max():.6f}]")
    #     print(f"\n  Top 10:")
    #     print(top_k_df.head(10).to_string(index=False))
        
    #     # High attention regions
    #     threshold_99 = stats['q99']
    #     high_attn_df = self.get_patches_above_threshold(threshold_99)
    #     print(f"\n[3] High Attention Patches (>= 99th percentile = {threshold_99:.6f}):")
    #     print(f"  Count: {len(high_attn_df)} ({len(high_attn_df)/len(self.attention)*100:.2f}%)")
        
    #     # Visualizations
    #     print(f"\n[4] Generating visualizations...")
    #     self.plot_distribution()
    #     self.plot_spatial_distribution()
    #     self.plot_top_k_heatmap(top_k)
        
    #     # Save outputs
    #     print(f"\n[5] Saving outputs...")
    #     self.save_top_k_patches(top_k)
        
    #     # Save statistics
    #     stats_path = self.output_dir / f"{self.wsi_name}_statistics.txt"
    #     with open(stats_path, 'w') as f:
    #         f.write(f"Attention Score Statistics - {self.wsi_name}\n")
    #         f.write("="*70 + "\n\n")
    #         for key, value in stats.items():
    #             f.write(f"{key:15s}: {value:.8f}\n")
    #     print(f"✓ Saved statistics: {stats_path}")
        
    #     print("\n" + "="*70)
    #     print("✓ REPORT COMPLETE!")
    #     print("="*70 + "\n")

    def generate_report(self, top_k=100):
            """Generate comprehensive analysis report"""
            print("\n" + "="*70)
            print(f"ATTENTION SCORE ANALYSIS REPORT - {self.wsi_name}")
            print("="*70)
            
            # Define the steps for the progress bar
            steps = [
                ("Calculating Statistics", self.get_statistics),
                ("Generating Distributions", lambda: self.plot_distribution(save=True)),
                ("Generating Spatial Plots", lambda: self.plot_spatial_distribution(save=True)),
                ("Generating Top-k Heatmap", lambda: self.plot_top_k_heatmap(k=top_k, save=True)),
                ("Saving Top-k Patches", lambda: self.save_top_k_patches(k=top_k))
            ]
    
            stats = None
            
            # Iterate through steps with a progress bar
            with tqdm(total=len(steps), desc="Processing Analysis", unit="step") as pbar:
                for description, func in steps:
                    pbar.set_description(f"Processing: {description}")
                    
                    # Execute the function
                    if description == "Calculating Statistics":
                        stats = func()
                    else:
                        func()
                    
                    pbar.update(1)
    
            # --- Print the Text Report (After progress bar completes) ---
            print("\n[Analysis Complete] Printing Summary:")
            
            # Statistics
            print(f"\n[1] Statistics:")
            print(f"  Total Patches: {len(self.attention)}")
            print(f"  Mean:          {stats['mean']:.6f}")
            print(f"  Std Dev:       {stats['std']:.6f}")
            print(f"  Min:           {stats['min']:.6f}")
            print(f"  Max:           {stats['max']:.6f}")
            
            # Top patches
            print(f"\n[2] Top-{top_k} Patches:")
            top_k_df = self.get_top_k_patches(top_k)
            print(f"  Attention range: [{top_k_df['attention'].min():.6f}, {top_k_df['attention'].max():.6f}]")
            print(f"\n  Top 10:")
            print(top_k_df.head(10).to_string(index=False))
            
            # Save statistics to file
            stats_path = self.output_dir / f"{self.wsi_name}_statistics.txt"
            with open(stats_path, 'w') as f:
                f.write(f"Attention Score Statistics - {self.wsi_name}\n")
                f.write("="*70 + "\n\n")
                for key, value in stats.items():
                    f.write(f"{key:15s}: {value:.8f}\n")
            print(f"\n✓ Saved full statistics to: {stats_path}")
            
            print("\n" + "="*70)
            print("✓ REPORT COMPLETE!")
            print("="*70 + "\n")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Analyze attention scores from WSI')
    
    # Required input file argument
    parser.add_argument('--input', 
                        type=str, 
                        required=True, 
                        help='Path to input attention file (.npy, .txt, .csv)')
    
    # Optional output directory argument
    parser.add_argument('--output', 
                        type=str, 
                        default=None, 
                        help='Path to output directory (optional)')
    
    # Optional top-k patches argument
    parser.add_argument('--top_k', 
                        type=int, 
                        default=100, 
                        help='Number of top patches to extract')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create analyzer with flexible input and output
    analyzer = AttentionAnalyzer(
        attention_file=args.input, 
        output_dir=args.output
    )
    
    # Generate report
    analyzer.generate_report(top_k=args.top_k)

if __name__ == '__main__':
    main()
    