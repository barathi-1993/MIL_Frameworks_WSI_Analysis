import os
import glob
import pandas as pd

dataset = "TCGA_COAD_READ"  # TCGA_NSCLC, STANFORD_793, TCGA_COAD_READ, CAMELYON16
root_dir = f'/data_64T_3/Raja/MUFASA/1.WSI_Classification/AUROC_Analysis/Collected_probabilities/{dataset}' 
preprocessors = ['CLAM', 'HISTOLAB', 'TRIDENT', 'MUFASA_SET_1']
models = ["MEAN_MIL","DeepAttn_MIL","CLAM_MB_MIL","DS_MIL","TRANS_MIL","DTFD_MIL","MHIM_MIL","ADD_MIL","DGR_MIL","ILRA_MIL","AC_MIL","TDA_MIL","MAMBA_MIL"]
output = f"/data_64T_3/Raja/MUFASA/1.WSI_Classification/AUROC_Analysis/Concatenated_probabilities_test/{dataset}"

def concatenate_fold_probabilities(root_dir, preprocessor, model_name, output):
    dataset_name = os.path.basename(root_dir)
    csv_filename = f'Detailed_Probabilities_{model_name}.csv'
    model_dir = os.path.join(root_dir, preprocessor, model_name)

    # Collect ALL subdirectories (supports fold_x AND seed_x)
    sub_dirs = sorted([d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d)) ])

    if not sub_dirs:
        print(f"⚠️  No subdirectories (fold/seed) found in: {model_dir}")
        return

    dfs = []
    for sub_dir in sub_dirs:
        # Check if CSV is directly in the sub_dir (Seed style) 
        # or inside a fold_1 subfolder (Camelyon16 edge case)
        csv_path = os.path.join(model_dir, sub_dir, csv_filename)
        
        # CAMELYON16 check: if csv not in seed_41/, check seed_41/fold_1/
        if not os.path.exists(csv_path):
            potential_nested = os.path.join(model_dir, sub_dir, 'fold_1', csv_filename)
            if os.path.exists(potential_nested):
                csv_path = potential_nested

        if not os.path.exists(csv_path):
            print(f"⚠️  Missing: {csv_filename} in {sub_dir} — skipping")
            continue

        df = pd.read_csv(csv_path)
        # Dynamic labeling: identifies if it's a seed or a fold
        label_type = 'seed' if 'seed' in sub_dir else 'fold'
        df[label_type] = sub_dir 
        
        dfs.append(df)
        print(f"✓ Loaded {len(df)} rows from {sub_dir}")

    if not dfs:
        print(f"⚠️  No CSVs successfully loaded for {model_name}. Aborting.")
        return

    # To remove duplicate entries
    # merged = pd.concat(dfs, ignore_index=True)
    # before = len(merged)

    # Deduplicate by slide_name
    # merged = merged.drop_duplicates(subset='slide_name', keep='first')
    # after = len(merged)

    # if before != after:
    #     print(f"⚠️  Removed {before - after} duplicate slide entries")

    # It includes duplicate samples when test is done with multiple seeds
    merged = pd.concat(dfs, ignore_index=True)
    after  = len(merged)

    os.makedirs(output, exist_ok=True)
    out_name = f'{dataset_name}_{model_name}_{preprocessor}_concatenated_probabilities.csv'
    out_path = os.path.join(output, out_name)
    merged.to_csv(out_path, index=False)

    print(f"✅ Saved merged {after} slides to: {out_path}")

if __name__ == '__main__': 
    print(f"Checking root directory: {root_dir}") 
    found_any = False
    for preprocessor in preprocessors:
        for model_name in models:
            model_dir = os.path.join(root_dir, preprocessor, model_name)
            
            # Debug: Print what we are checking
            if not os.path.isdir(model_dir):
                # Un-comment the line below if you want to see every failed path
                # print(f"🔍 Skipping (not found): {model_dir}")
                continue
            
            found_any = True
            print(f"\n🚀 Processing: {preprocessor} -> {model_name}")
            concatenate_fold_probabilities(root_dir, preprocessor, model_name, output)

    if not found_any:
        print("\n❌ ERROR: No valid model directories were found!")
        print(f"Please verify that {root_dir} contains folders named: {preprocessors}")

# def concatenate_fold_probabilities(root_dir, preprocessor, model_name, output):
#     """
#     Concatenates Detailed_Probabilities_<model_name>.csv from each fold,
#     removes duplicates by slide_name, and saves the merged file.

#     Args:
#         root_dir    : e.g. '/path/to/TCGA_CRC'
#         preprocessor   : e.g. 'MUFASA'
#         model_name  : e.g. 'AB_MIL'
#     """
#     dataset_name = os.path.basename(root_dir)          # TCGA_CRC
#     csv_filename = f'Detailed_Probabilities_{model_name}.csv'
#     model_dir    = os.path.join(root_dir, preprocessor, model_name)

#     # Collect all fold subdirectories (fold_1, fold_2, ...)
#     fold_dirs = sorted([
#         d for d in os.listdir(model_dir)
#         if os.path.isdir(os.path.join(model_dir, d))
#     ])

#     if not fold_dirs:
#         print(f"⚠️  No fold directories found in: {model_dir}")
#         return

#     dfs = []
#     for fold_dir in fold_dirs:
#         csv_path = os.path.join(model_dir, fold_dir, csv_filename)
#         if not os.path.exists(csv_path):
#             print(f"⚠️  Missing: {csv_path} — skipping")
#             continue
#         df = pd.read_csv(csv_path)
#         df['fold'] = fold_dir          # track which fold each row came from
#         dfs.append(df)
#         print(f"✓ Loaded {len(df)} rows from {fold_dir}")

#     if not dfs:
#         print("⚠️  No CSVs loaded. Aborting.")
#         return

#     merged = pd.concat(dfs, ignore_index=True)
#     before = len(merged)

#     # Deduplicate by slide_name — keep first occurrence
#     merged = merged.drop_duplicates(subset='slide_name', keep='first')
#     after  = len(merged)

#     if before != after:
#         print(f"⚠️  Removed {before - after} duplicate slide entries")

#     # Save
#     os.makedirs(output, exist_ok=True)   # create output dir if it doesn't exist
#     out_name = f'{dataset_name}_{model_name}_{preprocessor}_concatenated_probabilities.csv'
#     out_path = os.path.join(output, out_name)
#     merged.to_csv(out_path, index=False)

#     print(f"\n✓ Merged {after} slides from {len(dfs)} folds")
#     print(f"✓ Saved to: {out_path}")
#     return merged
 
# if __name__ == '__main__':
#     root_dir   = '/data_64T_3/Raja/MUFASA/1.WSI_Classification/AUROC_Analysis/Results/TCGA_NSCLC'
#     preprocessors = ['CLAM', 'HISTOLAB', 'TRIDENT', 'MUFASA_SET_1']
#     models     = ["MEAN_MIL","DeepAttn_MIL","CLAM_MB_MIL","DS_MIL","TRANS_MIL","DTFD_MIL","MHIM_MIL","ADD_MIL","DGR_MIL","ILRA_MIL","AC_MIL","TDA_MIL","MAMBA_MIL"]
#     output     = "/data_64T_3/Raja/MUFASA/1.WSI_Classification/AUROC_Analysis/Concatenated_probabilities"

#     for preprocessor in preprocessors:
#         for model_name in models:
#             model_dir = os.path.join(root_dir, preprocessor, model_name)
#             if not os.path.isdir(model_dir):
#                 continue
#             print(f"\n── {preprocessor} / {model_name} ──")
#             concatenate_fold_probabilities(root_dir, preprocessor, model_name, output)