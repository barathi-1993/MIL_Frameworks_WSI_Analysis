# # Extract both regions and top/bottom tiles
# python analyze_attention_score_on_ROI.py \
# --wsi /path/to/wsi/tif/svs \
# --attention /path/to/attention/file/txt \ 
# --extract_top_bottom \
# --top_k 100 --bottom_l 100 \
# --output /output/path

# python analyze_attention_score_on_ROI.py \
# --wsi /path/to/wsi/tif/svs \
# --attention /path/to/attention/file/txt \
# --extract_regions \
# --regions x11 y11 x12 y12 x21 y21 x22 y22 \ 
# --output /output/path

# python analyze_attention_score_on_ROI.py \
# --wsi /path/to/wsi/tif/svs \
# --attention /path/to/attention/file/txt \
# --extract_regions \
# --regions x11 y11 x12 y12 x21 y21 x22 y22 \
# --extract_top_bottom \
# --top_k 100 --bottom_l 100 \
# --output /output/path

import os
import argparse
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
import imageio

class TileExtractor:
    def __init__(self, wsi_path, attention_file, output_dir=None):
        """
        Initialize the TileExtractor
        
        Args:
            wsi_path (str): Path to whole slide image
            attention_file (str): Path to attention score text file
            output_dir (str, optional): Directory to save output files
        """
        # Image reading setup (using previous implementation)
        self._setup_image_reading(wsi_path)
        
        # Set output directory
        self.output_dir = Path(output_dir) if output_dir else Path(f"extracted_tiles_{self.wsi_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load attention scores
        self.attention_df = self._load_attention_file(attention_file)
        
        # Determine tile size
        self.tile_size = self._determine_tile_size()
        
        # Calculate total tiles in WSI
        self.total_x_tiles = int(self.wsi_dimensions[0] / self.tile_size)
        self.total_y_tiles = int(self.wsi_dimensions[1] / self.tile_size)

    def _load_attention_file(self, attention_file):
        """
        Load attention scores from text file
        
        Returns:
            pd.DataFrame: Dataframe with x, y coordinates and attention scores
        """
        df = pd.read_csv(attention_file, 
                         sep='\s+', 
                         comment='#', 
                         header=None, 
                         names=['x_coord', 'y_coord', 'attention_score'])
        return df

    def _determine_tile_size(self):
        """
        Determine tile size from unique coordinate differences
        
        Returns:
            int: Tile size (e.g., 256, 512, 1024)
        """
        unique_x = sorted(np.unique(self.attention_df['x_coord']))
        if len(unique_x) > 1:
            return unique_x[1] - unique_x[0]
        else:
            return 512  # Default tile size if unable to determine
    
    def _setup_image_reading(self, wsi_path):
        """
        Set up image reading method
        
        Args:
            wsi_path (str): Path to whole slide image
        """
        # Try OpenSlide first
        try:
            import openslide
            try:
                self.wsi = openslide.OpenSlide(wsi_path)
                self.reading_method = 'openslide'
            except:
                self.wsi = None
        except ImportError:
            self.wsi = None
        
        # Fallback to imageio or PIL if OpenSlide fails
        if self.wsi is None:
            try:
                self.wsi_image = imageio.imread(wsi_path)
                self.reading_method = 'imageio'
                self.wsi_dimensions = self.wsi_image.shape[:2][::-1]  # (width, height)
            except:
                try:
                    from PIL import Image
                    self.wsi_image = Image.open(wsi_path)
                    self.reading_method = 'pil'
                    self.wsi_dimensions = self.wsi_image.size
                except Exception as e:
                    raise ValueError(f"Unable to read image file: {e}")
        else:
            self.wsi_dimensions = self.wsi.dimensions
        
        self.wsi_name = os.path.splitext(os.path.basename(wsi_path))[0]
    
    def extract_top_bottom_tiles(self, top_k=100, bottom_l=100):
        """
        Extract top k and bottom l tiles based on entire attention scores
        
        Args:
            top_k (int): Number of top tiles to extract
            bottom_l (int): Number of bottom tiles to extract
        
        Returns:
            tuple: DataFrames for top and bottom tiles
        """
        # Sort attention scores
        sorted_attention = self.attention_df.sort_values('attention_score')
        
        # Extract bottom and top tiles
        bottom_tiles = sorted_attention.head(bottom_l)
        top_tiles = sorted_attention.tail(top_k)
        
        # Create output directories
        top_tiles_dir = self.output_dir / "top_100_tiles"
        bottom_tiles_dir = self.output_dir / "bottom_100_tiles"
        top_tiles_dir.mkdir(parents=True, exist_ok=True)
        bottom_tiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract and save top and bottom tiles
        self._extract_tiles_from_dataframe(top_tiles, top_tiles_dir, "top")
        self._extract_tiles_from_dataframe(bottom_tiles, bottom_tiles_dir, "bottom")
        
        # Save CSV files
        top_tiles.to_csv(self.output_dir / "top_100_attention.csv", index=False)
        bottom_tiles.to_csv(self.output_dir / "bottom_100_attention.csv", index=False)
        
        return top_tiles, bottom_tiles
    
    def extract_tiles_from_regions(self, regions):
        """
        Extract tiles from specified regions of interest
        
        Args:
            regions (list): List of region coordinates 
                            [[x1,y1,x2,y2], [x11,y11,x12,y12], ...]
        
        Returns:
            list: Extracted tile information
        """
        all_extracted_regions = []
        
        # for region_idx, (x1, y1, x2, y2) in enumerate(regions):
        for region_idx, (x1, y1, x2, y2) in enumerate(tqdm(regions, desc="Processing Regions")):
            # Create region-specific output directory
            region_name = f"region_{x1}_{y1}_{x2}_{y2}_tiles"
            region_output_dir = self.output_dir / region_name
            region_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Ensure coordinates are within WSI dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(x2, self.wsi_dimensions[0])
            y2 = min(y2, self.wsi_dimensions[1])
            
            # Find tiles within this region
            region_tiles = self._find_tiles_in_region(x1, y1, x2, y2)
            
            # Save region attention CSV
            region_attention_df = pd.DataFrame(region_tiles)
            region_attention_csv = self.output_dir / f"region_{x1}_{y1}_{x2}_{y2}_attention.csv"
            region_attention_df.to_csv(region_attention_csv, index=False)
            
            # Extract tiles
            extracted_region_tiles = self._extract_tiles_from_dataframe(
                region_attention_df, 
                region_output_dir, 
                f"region_{x1}_{y1}_{x2}_{y2}"
            )
            
            all_extracted_regions.append({
                'region': (x1, y1, x2, y2),
                'tiles': extracted_region_tiles
            })
        
        return all_extracted_regions

    def _find_tiles_in_region(self, x1, y1, x2, y2):
        """
        Find tiles within a specific region with flexible coordinate matching
        
        Args:
            x1, y1 (int): Top-left coordinates
            x2, y2 (int): Bottom-right coordinates
        
        Returns:
            pd.DataFrame: Tiles within the region with consistent column names
        """
        # Ensure coordinates are within WSI dimensions
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(x2, self.wsi_dimensions[0])
        y2 = min(y2, self.wsi_dimensions[1])
        
        # Find nearest valid tile coordinates
        x_start = self._find_nearest_tile_coordinate(x1, self.attention_df['x_coord'])
        y_start = self._find_nearest_tile_coordinate(y1, self.attention_df['y_coord'])
        
        # Find tiles within this region
        region_tiles = []
        missing_tiles = []
        
        # Calculate tile coordinates within the region
        for x in range(x_start, x2, self.tile_size):
            for y in range(y_start, y2, self.tile_size):
                # Check if this tile exists in attention scores
                tile_coords = self.attention_df[
                    (self.attention_df['x_coord'] == x) & 
                    (self.attention_df['y_coord'] == y)
                ]
                
                if not tile_coords.empty:
                    # Calculate tile indices
                    x_tile_idx = int(x / self.tile_size)
                    y_tile_idx = int(y / self.tile_size)
                    
                    region_tiles.append({
                        'x_coord': x,
                        'y_coord': y,
                        'attention_score': tile_coords['attention_score'].values[0],
                        'x_tile_idx': x_tile_idx,
                        'y_tile_idx': y_tile_idx,
                        'total_x_tiles': self.total_x_tiles,
                        'total_y_tiles': self.total_y_tiles
                    })
                else:
                    # Log missing tiles
                    missing_tiles.append({
                        'x_coord': x,
                        'y_coord': y,
                        'reason': 'No attention score available'
                    })
        
        # Log missing tiles if any
        if missing_tiles:
            missing_tiles_path = self.output_dir / f"missing_tiles_region_{x1}_{y1}_{x2}_{y2}.txt"
            with open(missing_tiles_path, 'w') as f:
                f.write("Missing Tiles in Region:\n")
                f.write("=" * 50 + "\n")
                f.write("x_coord\ty_coord\treason\n")
                for tile in missing_tiles:
                    f.write(f"{tile['x_coord']}\t{tile['y_coord']}\t{tile['reason']}\n")
            
            print(f"Missing tiles logged: {missing_tiles_path}")
        
        # Convert to DataFrame with consistent column names
        return pd.DataFrame(region_tiles)

    def _find_nearest_tile_coordinate(self, target_coord, available_coords):
        """
        Find the nearest valid tile coordinate
        
        Args:
            target_coord (int): Target coordinate
            available_coords (pd.Series): Available coordinates
        
        Returns:
            int: Nearest valid tile coordinate
        """
        # Find coordinates greater than or equal to target
        valid_coords = available_coords[available_coords >= target_coord]
        
        if valid_coords.empty:
            # If no coordinates are greater, return the maximum available coordinate
            return available_coords.max()
        
        # Return the minimum coordinate that is greater than or equal to target
        return valid_coords.min()
    
    # def _extract_tiles_from_dataframe(self, dataframe, output_dir, prefix):
    #     """
    #     Extract tiles from a DataFrame of coordinates
        
    #     Args:
    #         dataframe (pd.DataFrame): DataFrame with coordinates
    #         output_dir (Path): Directory to save tiles
    #         prefix (str): Prefix for tile filenames
        
    #     Returns:
    #         list: List of extracted tile information
    #     """
    #     extracted_tiles = []
        
    #     for _, row in dataframe.iterrows():
    #         x, y = int(row['x_coord']), int(row['y_coord'])
            
    #         try:
    #             # Read tile from WSI
    #             tile = self._read_tile(x, y, self.tile_size)
                
    #             # Construct filename
    #             filename = (f"{self.wsi_name}_512_w{y}_107_186_h{x}_151_160_"
    #                         f"{prefix}.png")
    #             tile_path = output_dir / filename
                
    #             # Save tile
    #             if self.reading_method in ['openslide', 'pil']:
    #                 tile.save(tile_path)
    #             else:
    #                 cv2.imwrite(str(tile_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                
    #             extracted_tiles.append({
    #                 'filename': filename,
    #                 'x_coord': x,
    #                 'y_coord': y,
    #                 'attention_score': row['attention_score']
    #             })
            
    #         except Exception as e:
    #             print(f"Error extracting tile at ({x}, {y}): {e}")
        
    #     return extracted_tiles

    def _extract_tiles_from_dataframe(self, dataframe, output_dir, prefix):
        """
        Extract tiles from a DataFrame of coordinates
        """
        extracted_tiles = []
             
        # desc provides a label, total ensures the bar knows the 100% mark
        print(f"Extracting {len(dataframe)} tiles to {output_dir}...")
        for _, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc=f"Saving {prefix}"):
            x, y = int(row['x_coord']), int(row['y_coord'])
                
            try:
                # Read tile from WSI
                tile = self._read_tile(x, y, self.tile_size)
                    
                # Construct filename
                filename = (f"{self.wsi_name}_512_w{y}_107_186_h{x}_151_160_"
                                f"{prefix}.png")
                tile_path = output_dir / filename
                    
                # Save tile
                if self.reading_method in ['openslide', 'pil']:
                    tile.save(tile_path)
                else:
                    cv2.imwrite(str(tile_path), cv2.cvtColor(tile, cv2.COLOR_RGB2BGR))
                    
                extracted_tiles.append({
                        'filename': filename,
                        'x_coord': x,
                        'y_coord': y,
                        'attention_score': row['attention_score']
                    })
                
            except Exception as e:
                print(f"Error extracting tile at ({x}, {y}): {e}")
            
        return extracted_tiles

    def _read_tile(self, x, y, tile_size):
        """
        Read a tile from the image using the appropriate method
        
        Args:
            x (int): X coordinate
            y (int): Y coordinate
            tile_size (int): Size of the tile
        
        Returns:
            Image or numpy array of the tile
        """
        if self.reading_method == 'openslide':
            tile = self.wsi.read_region((x, y), 0, (tile_size, tile_size))
            return tile.convert('RGB')
        elif self.reading_method == 'imageio':
            return self.wsi_image[y:y+tile_size, x:x+tile_size]
        elif self.reading_method == 'pil':
            return self.wsi_image.crop((x, y, x+tile_size, y+tile_size))
        else:
            raise ValueError(f"Unsupported reading method: {self.reading_method}")

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Extract tiles from Whole Slide Image')
    
    # Required arguments
    parser.add_argument('--wsi', 
                        type=str, 
                        required=True, 
                        help='Path to whole slide image')
    
    parser.add_argument('--attention', 
                        type=str, 
                        required=True, 
                        help='Path to attention scores file')
    
    # Optional arguments
    parser.add_argument('--output', 
                        type=str, 
                        default=None, 
                        help='Output directory (optional)')
    
    # Extraction type arguments
    parser.add_argument('--extract_regions', 
                        action='store_true', 
                        help='Extract tiles from specified regions')
    
    parser.add_argument('--extract_top_bottom', 
                        action='store_true', 
                        help='Extract top and bottom tiles')
    
    parser.add_argument('--regions', 
                        type=int, 
                        nargs='*', 
                        help='Regions of interest as x1 y1 x2 y2 for each region')
    
    parser.add_argument('--top_k', 
                        type=int, 
                        default=100, 
                        help='Number of top tiles to extract')
    
    parser.add_argument('--bottom_l', 
                        type=int, 
                        default=100, 
                        help='Number of bottom tiles to extract')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Process regions if provided
    if args.regions:
        try:
            regions = [list(args.regions[i:i+4]) 
                       for i in range(0, len(args.regions), 4)]
        except ValueError:
            print("Error: Regions must be provided as sets of 4 integers")
            return
    else:
        regions = []
    
    # Initialize extractor
    extractor = TileExtractor(
        wsi_path=args.wsi, 
        attention_file=args.attention, 
        output_dir=args.output
    )
    
    # Perform extraction based on arguments
    if not (args.extract_regions or args.extract_top_bottom):
        print("Please specify --extract_regions or --extract_top_bottom")
        return
    
    if args.extract_regions and regions:
        extractor.extract_tiles_from_regions(regions)
    
    if args.extract_top_bottom:
        extractor.extract_top_bottom_tiles(
            top_k=args.top_k, 
            bottom_l=args.bottom_l
        )

if __name__ == '__main__':
    main()