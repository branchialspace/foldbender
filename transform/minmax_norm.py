# Minmax normalization for x and edge_attr
# Rounding x, edge_attr and atom_coords
import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

def minmax_norm(input_dir):
    stats = os.path.join(os.path.dirname(input_dir), f"{os.path.basename(input_dir)}_stats.csv")
    norm_stats = os.path.join(os.path.dirname(input_dir), f"{os.path.basename(input_dir)}_norm_stats.csv")

    # Initialize global min and max arrays
    global_x_min = None
    global_x_max = None
    global_edge_attr_min = None
    global_edge_attr_max = None

    # First pass to find the global min and max for each feature
    for file_name in tqdm(os.listdir(input_dir), desc="Finding global min and max for each feature"):
        if file_name.endswith('.pt'):
            data_path = os.path.join(input_dir, file_name)
            data_object = torch.load(data_path)

            # Convert to float32 for min/max calculation
            x_min = data_object.x.float().min(dim=0).values.cpu().numpy()
            x_max = data_object.x.float().max(dim=0).values.cpu().numpy()
            edge_attr_min = data_object.edge_attr.float().min(dim=0).values.cpu().numpy()
            edge_attr_max = data_object.edge_attr.float().max(dim=0).values.cpu().numpy()

            if global_x_min is None:
                global_x_min, global_x_max = x_min, x_max
                global_edge_attr_min, global_edge_attr_max = edge_attr_min, edge_attr_max
            else:
                global_x_min = np.minimum(global_x_min, x_min)
                global_x_max = np.maximum(global_x_max, x_max)
                global_edge_attr_min = np.minimum(global_edge_attr_min, edge_attr_min)
                global_edge_attr_max = np.maximum(global_edge_attr_max, edge_attr_max)

    feature_names_x = [f'x_feature_{i}' for i in range(len(global_x_min))]
    feature_names_edge_attr = [f'edge_attr_feature_{i}' for i in range(len(global_edge_attr_min))]

    scaling_parameters_df = pd.DataFrame({
        'feature': feature_names_x + feature_names_edge_attr,
        'min': np.concatenate((global_x_min, global_edge_attr_min)),
        'max': np.concatenate((global_x_max, global_edge_attr_max))
    })
    scaling_parameters_df.to_csv(stats, index=False)

    accumulated_x_mins = []
    accumulated_x_maxs = []
    accumulated_edge_attr_mins = []
    accumulated_edge_attr_maxs = []

    # Second pass to scale the data using the global min and max
    for file_name in tqdm(os.listdir(input_dir), desc="Scaling the data using the global min and max"): 
        if file_name.endswith('.pt'):
            data_path = os.path.join(input_dir, file_name)
            data_object = torch.load(data_path)

            # Perform normalization in float32
            x_norm = (data_object.x.float() - torch.tensor(global_x_min, dtype=torch.float32)) / \
                     (torch.tensor(global_x_max, dtype=torch.float32) - torch.tensor(global_x_min, dtype=torch.float32))
            edge_attr_norm = (data_object.edge_attr.float() - torch.tensor(global_edge_attr_min, dtype=torch.float32)) / \
                             (torch.tensor(global_edge_attr_max, dtype=torch.float32) - torch.tensor(global_edge_attr_min, dtype=torch.float32))

            # Handle NaN values
            x_norm[torch.isnan(x_norm)] = 0.0
            edge_attr_norm[torch.isnan(edge_attr_norm)] = 0.0

            # Round off
            x_norm = torch.round(x_norm * 10000000) / 10000000
            edge_attr_norm = torch.round(edge_attr_norm * 10000000) / 10000000
            data_object.atom_coords = torch.round(data_object.atom_coords.float() * 10000000) / 10000000

            accumulated_x_mins.append(data_object.x.min(dim=0).values)
            accumulated_x_maxs.append(data_object.x.max(dim=0).values)
            accumulated_edge_attr_mins.append(data_object.edge_attr.min(dim=0).values)
            accumulated_edge_attr_maxs.append(data_object.edge_attr.max(dim=0).values)
            
            # Convert back to float16
            data_object.x = x_norm.to(dtype=torch.float16)
            data_object.edge_attr = edge_attr_norm.to(dtype=torch.float16)
            data_object.atom_coords = data_object.atom_coords.to(dtype=torch.float16)

            torch.save(data_object, data_path)

    global_rounded_x_min = torch.stack(accumulated_x_mins).min(dim=0).values.numpy()
    global_rounded_x_max = torch.stack(accumulated_x_maxs).max(dim=0).values.numpy()
    global_rounded_edge_attr_min = torch.stack(accumulated_edge_attr_mins).min(dim=0).values.numpy()
    global_rounded_edge_attr_max = torch.stack(accumulated_edge_attr_maxs).max(dim=0).values.numpy()

    norm_rounded_stats_df = pd.DataFrame({
        'feature': feature_names_x + feature_names_edge_attr,
        'rounded_min': np.round(np.concatenate((global_rounded_x_min, global_rounded_edge_attr_min)), 6),
        'rounded_max': np.round(np.concatenate((global_rounded_x_max, global_rounded_edge_attr_max)), 6)
    })
    norm_rounded_stats_df.to_csv(norm_stats, index=False)

if __name__ == "__main__":

    input_dir = '/content/drive/MyDrive/protein-DATA/prot-sample'
    
    minmax_norm(input_dir)
