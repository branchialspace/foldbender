import os
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import Data

input_dir = '/content/drive/MyDrive/protein-DATA/prot-sample'
output_dir = '/content/drive/MyDrive/protein-DATA/sample-normalized'
stats = '/content/drive/MyDrive/protein-DATA/minmax-stats.csv'
norm_stats = '/content/drive/MyDrive/protein-DATA/norm-stats.csv'

# Initialize global min and max arrays
global_x_min = None
global_x_max = None
global_edge_attr_min = None
global_edge_attr_max = None

# Initialize global normalized and rounded min and max
global_norm_x_min = None
global_norm_x_max = None
global_norm_edge_attr_min = None
global_norm_edge_attr_max = None

# First pass to find the global min and max for each feature
for file_name in os.listdir(input_dir):
    if file_name.endswith('.pt'):
        data_path = os.path.join(input_dir, file_name)
        data_dict = torch.load(data_path)
        data_key = file_name[:-3]
        data_object = data_dict[data_key]

        if global_x_min is None:
            global_x_min = data_object.x.min(dim=0).values.cpu().numpy()
            global_x_max = data_object.x.max(dim=0).values.cpu().numpy()
            global_edge_attr_min = data_object.edge_attr.min(dim=0).values.cpu().numpy()
            global_edge_attr_max = data_object.edge_attr.max(dim=0).values.cpu().numpy()
        else:
            global_x_min = np.minimum(global_x_min, data_object.x.min(dim=0).values.cpu().numpy())
            global_x_max = np.maximum(global_x_max, data_object.x.max(dim=0).values.cpu().numpy())
            global_edge_attr_min = np.minimum(global_edge_attr_min, data_object.edge_attr.min(dim=0).values.cpu().numpy())
            global_edge_attr_max = np.maximum(global_edge_attr_max, data_object.edge_attr.max(dim=0).values.cpu().numpy())

# Save the global min and max values
feature_names_x = [f'x_feature_{i}' for i in range(global_x_min.shape[0])]
feature_names_edge_attr = [f'edge_attr_feature_{i}' for i in range(global_edge_attr_min.shape[0])]
scaling_parameters_df = pd.DataFrame({
    'feature': feature_names_x + feature_names_edge_attr,
    'min': np.concatenate((global_x_min, global_edge_attr_min)),
    'max': np.concatenate((global_x_max, global_edge_attr_max))
})
scaling_parameters_df.to_csv(stats, index=False)

# Second pass to scale the data and update the normalized and rounded global stats
for file_name in os.listdir(input_dir):
    if file_name.endswith('.pt'):
        data_path = os.path.join(input_dir, file_name)
        data_dict = torch.load(data_path)
        data_key = file_name[:-3]
        data_object = data_dict[data_key]

        # Normalize features
        x_min_tensor = torch.tensor(global_x_min, dtype=torch.float32)
        x_max_tensor = torch.tensor(global_x_max, dtype=torch.float32)
        edge_attr_min_tensor = torch.tensor(global_edge_attr_min, dtype=torch.float32)
        edge_attr_max_tensor = torch.tensor(global_edge_attr_max, dtype=torch.float32)
        data_object.x = (data_object.x - x_min_tensor) / (x_max_tensor - x_min_tensor)
        data_object.edge_attr = (data_object.edge_attr - edge_attr_min_tensor) / (edge_attr_max_tensor - edge_attr_min_tensor)

        # Round normalized values
        data_object.x = torch.round(data_object.x * 10000) / 10000
        data_object.edge_attr = torch.round(data_object.edge_attr * 10000) / 10000

        # Update the dictionary with the scaled and rounded Data object
        data_dict[data_key] = data_object
        torch.save(data_dict, os.path.join(output_dir, file_name))

        # Update global normalized and rounded min and max
        if global_norm_x_min is None:
            global_norm_x_min = data_object.x.min(dim=0).values.cpu().numpy()
            global_norm_x_max = data_object.x.max(dim=0).values.cpu().numpy()
            global_norm_edge_attr_min = data_object.edge_attr.min(dim=0).values.cpu().numpy()
            global_norm_edge_attr_max = data_object.edge_attr.max(dim=0).values.cpu().numpy()
        else:
            global_norm_x_min = np.minimum(global_norm_x_min, data_object.x.min(dim=0).values.cpu().numpy())
            global_norm_x_max = np.maximum(global_norm_x_max, data_object.x.max(dim=0).values.cpu().numpy())
            global_norm_edge_attr_min = np.minimum(global_norm_edge_attr_min, data_object.edge_attr.min(dim=0).values.cpu().numpy())
            global_norm_edge_attr_max = np.maximum(global_norm_edge_attr_max, data_object.edge_attr.max(dim=0).values.cpu().numpy())

# Round the global normalized min and max to four decimal places
global_norm_x_min = np.round(global_norm_x_min, 4)
global_norm_x_max = np.round(global_norm_x_max, 4)
global_norm_edge_attr_min = np.round(global_norm_edge_attr_min, 4)
global_norm_edge_attr_max = np.round(global_norm_edge_attr_max, 4)

# Create a DataFrame for storing feature-wise normalized and rounded min and max
norm_rounded_stats_df = pd.DataFrame({
    'feature': feature_names_x + feature_names_edge_attr,
    'rounded_min': np.concatenate((global_norm_x_min, global_norm_edge_attr_min)),
    'rounded_max': np.concatenate((global_norm_x_max, global_norm_edge_attr_max))
})
norm_rounded_stats_df.to_csv(norm_stats, index=False)
