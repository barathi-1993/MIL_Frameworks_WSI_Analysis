import os
import random
import sys
import ast
import argparse

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
grandparent_dir = os.path.abspath(os.path.join(current_dir, "../"))
sys.path.append(grandparent_dir)

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score, silhouette_samples

import torch
from torch.utils.data import DataLoader

# Utils imports (Keep as is)
from utils.loop_utils import val_loop, clam_val_loop, ds_val_loop, dtfd_val_loop, ptc_val_loop, deepattn_val_loop
from utils.model_utils import get_model_from_yaml, get_criterion
from utils.yaml_utils import read_yaml
from utils.wsi_utils import WSI_Dataset_Test, CDP_MIL_WSI_Dataset, LONG_MIL_WSI_Dataset

import warnings
warnings.filterwarnings('ignore')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"🔒 Seed set to {seed} for reproducibility.")

def draw_tsne(feature_tensor, label_tensor, id2class, save_path, fig_size=(10, 8), seed=42): 
    # Handle logic: If id2class is None/Empty, generate it automatically
    unique_labels = np.unique(label_tensor)
    if not id2class:
        id2class = {int(label): f"Class_{label}" for label in unique_labels}
    
    # Ensure keys are integers for correct mapping
    id2class = {int(k): v for k, v in id2class.items()}

    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    perplexity = min(30, feature_tensor.shape[0] - 1)
    tsne = TSNE(perplexity=perplexity, n_components=2, random_state=seed)
    tsne_result = tsne.fit_transform(feature_tensor)
    
    plt.figure(figsize=fig_size)
    for label_id in unique_labels:
        # Only plot if this label exists in our datda
        indices = np.where(label_tensor == int(label_id))[0]
        # Get class name from dictionary
        class_name = id2class.get(int(label_id), f"Class_{label_id}")
        
        plt.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                    label=class_name, alpha=0.7, s=50)
    
    plt.legend(frameon=True, loc='best')
    plt.xlabel('t-SNE Component 1', fontsize=12)
    plt.ylabel('t-SNE Component 2', fontsize=12)
    plt.title('t-SNE Visualization', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
    plt.close()

def draw_silhouette(feature_tensor, label_tensor, save_path, id2class=None):
    """
    Calculates the Silhouette Score and plots the Silhouette Analysis.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # 1. Calculate the overall mean silhouette score
    if len(np.unique(label_tensor)) < 2:
        print("Skipping Silhouette plot: Only 1 class present in data.")
        return

    silhouette_avg = silhouette_score(feature_tensor, label_tensor)
    print(f"For n_clusters = {len(np.unique(label_tensor))}, The average silhouette_score is : {silhouette_avg:.4f}")
    
    # 2. Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(feature_tensor, label_tensor)

    fig, ax1 = plt.subplots(1, 1)
    fig.set_size_inches(10, 7)

    # The silhouette coefficient can range from -1, 1
    ax1.set_xlim([-0.1, 1])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette plots of individual clusters
    ax1.set_ylim([0, len(feature_tensor) + (len(np.unique(label_tensor)) + 1) * 10])

    y_lower = 10
    unique_labels = np.unique(label_tensor)
    
    for i in unique_labels:
        # Aggregate the silhouette scores for samples belonging to cluster i, and sort them
        ith_cluster_silhouette_values = sample_silhouette_values[label_tensor == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / len(unique_labels))
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                          0, ith_cluster_silhouette_values,
                          facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers
        label_name = id2class[int(i)] if id2class else f"Class {int(i)}"
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, label_name)

        y_lower = y_upper + 10  # 10 for the 0 samples

    ax1.set_title("Silhouette Plot for Preprocessing Comparison")
    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_ylabel("Cluster label")

    # The vertical line for average silhouette score of all the values
    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

    # Save
    silhouette_save_path = save_path.replace('.png', '_silhouette.png').replace('.jpg', '_silhouette.jpg')
    plt.savefig(silhouette_save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Silhouette plot saved at {silhouette_save_path}")

def main(args):
    seed_everything(args.seed) 
    yaml_path = args.yaml_path
    yaml_args = read_yaml(yaml_path)
    model_name = yaml_args.General.MODEL_NAME
    mil_model = get_model_from_yaml(yaml_args)
    print(f"Model name: {model_name}")
    num_classes = yaml_args.General.num_classes
    test_dataset_csv = args.test_dataset_csv
    print(f"Dataset csv path: {test_dataset_csv}")
    data_split = args.data_split
    print(f"Data split to evaluate: {data_split}")
    
    # Dataset Loading
    if model_name == 'CDP_MIL':
        raise NotImplementedError("CDP_MIL model is not supported for feature map visualization now.")
    elif model_name == 'LONG_MIL':
        LONG_MIL_WSI_Dataset(test_dataset_csv,yaml_args.Dataset.h5_csv_path,'test')
        
    test_ds = WSI_Dataset_Test(test_dataset_csv, args, data_split)
    test_dataloader = DataLoader(test_ds, batch_size=1, shuffle=False)
    model_weight_path = args.ckpt_path
    print(f"Model weight path: {model_weight_path}")
    
    device = torch.device(f'cuda:{yaml_args.General.device}')
    criterion = get_criterion(yaml_args.Model.criterion)
    
    # Model Loading
    if yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        classifier, attention, dimReduction, attCls = get_model_from_yaml(yaml_args)
        state_dict = torch.load(model_weight_path, weights_only=True)
        classifier.load_state_dict(state_dict['classifier'])
        attention.load_state_dict(state_dict['attention'])
        dimReduction.load_state_dict(state_dict['dimReduction'])
        attCls.load_state_dict(state_dict['attCls'])
        model_list = [classifier, attention, dimReduction, attCls]
        model_list = [model.to(device).eval() for model in model_list]
    elif yaml_args.General.MODEL_NAME == 'PTC_MIL':
        # Instantiate the model using the class structure you provided
        # Note: ViT_PMT_CLU requires 'n_classes' and 'args' (from yaml)
        from modules.PTC_MIL.model_vit_pmt_clu import ViT_PMT_CLU
        mil_model = ViT_PMT_CLU(n_classes=num_classes, args=yaml_args.Model)        
        # Load weights
        state_dict = torch.load(model_weight_path)
        mil_model.load_state_dict(state_dict)
        mil_model.to(device).eval()
    else:
        mil_model = mil_model.to(device)
        mil_model.load_state_dict(torch.load(model_weight_path, weights_only=True))

    # Feature Extraction Loop
    if yaml_args.General.MODEL_NAME == 'CLAM_MB_MIL' or yaml_args.General.MODEL_NAME == 'CLAM_SB_MIL':
        bag_weight = yaml_args.Model.bag_weight
        WSI_features = clam_val_loop(device, num_classes, mil_model, test_dataloader, criterion, bag_weight, retrun_WSI_feature=True)
    elif yaml_args.General.MODEL_NAME == 'DS_MIL':
        WSI_features = ds_val_loop(device, num_classes, mil_model, test_dataloader, criterion, retrun_WSI_feature=True)
    elif yaml_args.General.MODEL_NAME == 'DTFD_MIL':
        WSI_features = dtfd_val_loop(device, num_classes, model_list, test_dataloader, criterion, yaml_args.Model.num_Group, yaml_args.Model.grad_clipping, yaml_args.Model.distill, yaml_args.Model.total_instance, retrun_WSI_feature=True)
    elif yaml_args.General.MODEL_NAME == 'PTC_MIL':
        WSI_features = ptc_val_loop(device, num_classes, mil_model, test_dataloader, criterion, retrun_WSI_feature=True, return_WSI_attn=False)
    elif yaml_args.General.MODEL_NAME == 'DeepAttn_MIL':
        WSI_features = deepattn_val_loop(device, num_classes, mil_model, test_dataloader, criterion, retrun_WSI_feature=True, return_WSI_attn=False)        
        # ------------------------------------------
    else:
        WSI_features = val_loop(device, num_classes, mil_model, test_dataloader, criterion, retrun_WSI_feature=True)

    WSI_labels = np.array(test_ds.labels_list)
    WSI_labels = WSI_labels.astype(int) 

    # 1. Ensure WSI_features is a numpy array (it might still be a Torch Tensor)
    if isinstance(WSI_features, torch.Tensor):
        WSI_features = WSI_features.cpu().numpy()
    
    # 2. FIX: If features are flattened (1D) but we have multiple labels (slides), 
    # reshape them back to (Num_Slides, Feature_Dim) to fix the "Expected 2D array" error.
    num_slides = len(WSI_labels)
    if len(WSI_features.shape) == 1 and num_slides > 1:
        print(f"Warning: Features were flattened (shape {WSI_features.shape}). Reshaping based on {num_slides} slides...")
        # Automatically infer the embedding dim (e.g., 512 or 1024)
        WSI_features = WSI_features.reshape(num_slides, -1)
        print(f"New Feature Shape: {WSI_features.shape}")
    
    # 1. Parse id2class from string safely
    if args.id2class:
        try:
            id2class_dict = ast.literal_eval(args.id2class)
        except:
            print("Warning: Could not parse id2class string. Using default labels.")
            id2class_dict = None
    else:
        id2class_dict = None

    # 2. Draw t-SNE
    draw_tsne(WSI_features, WSI_labels, id2class_dict, args.save_path, (5,4)) 
    print(f"TSNE plot saved at {args.save_path}")

    # 3. Draw Silhouette
    # draw_silhouette(WSI_features, WSI_labels, args.save_path, id2class=id2class_dict)

if __name__ == "__main__":      
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--yaml_path', type=str, default='', help='path to yaml file') 
    argparser.add_argument('--ckpt_path', type=str, default='', help='path to pretrained weights')
    argparser.add_argument('--save_path', type=str, default='', help='path to save the model')
    argparser.add_argument('--id2class', type=str, default='', help='str type dictionary mapping label ids to class names')
    argparser.add_argument('--test_dataset_csv', type=str, default='', help='path to dataset csv file')
    argparser.add_argument('--data_split', type=str, default='', help='train/val/test')
    argparser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility') 
    args = argparser.parse_args()
    main(args)