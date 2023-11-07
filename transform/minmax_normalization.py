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

# Initialize arrays to hold the minimum and maximum of normalized values
norm_min_x = []
norm_max_x = []
norm_min_edge_attr = []
norm_max_edge_attr = []

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

# Create a DataFrame for storing feature-wise min and max
feature_names_x = [f'x_feature_{i}' for i in range(global_x_min.shape[0])]
feature_names_edge_attr = [f'edge_attr_feature_{i}' for i in range(global_edge_attr_min.shape[0])]

scaling_parameters_df = pd.DataFrame({
    'feature': feature_names_x + feature_names_edge_attr,
    'min': np.concatenate((global_x_min, global_edge_attr_min)),
    'max': np.concatenate((global_x_max, global_edge_attr_max))
})

scaling_parameters_df.to_csv(stats, index=False)

# Second pass to scale, round the data using the global min and max, and save the output
for file_name in os.listdir(input_dir):
    if file_name.endswith('.pt'):
        data_path = os.path.join(input_dir, file_name)
        data_dict = torch.load(data_path)
        data_key = file_name[:-3]
        data_object = data_dict[data_key]

        # Normalize x and edge_attr
        data_object.x = (data_object.x - global_x_min) / (global_x_max - global_x_min)
        data_object.edge_attr = (data_object.edge_attr - global_edge_attr_min) / (global_edge_attr_max - global_edge_attr_min)

        # Round x and edge_attr before saving to the output file
        data_object.x = torch.round(data_object.x * 10000) / 10000
        data_object.edge_attr = torch.round(data_object.edge_attr * 10000) / 10000

        # Save the modified data dictionary with rounded values
        output_path = os.path.join(output_dir, file_name)
        torch.save(data_dict, output_path)

        # Update min and max for normalized and rounded values
        norm_min_x.append(data_object.x.min(dim=0).values.cpu().numpy())
        norm_max_x.append(data_object.x.max(dim=0).values.cpu().numpy())
        norm_min_edge_attr.append(data_object.edge_attr.min(dim=0).values.cpu().numpy())
        norm_max_edge_attr.append(data_object.edge_attr.max(dim=0).values.cpu().numpy())

# Calculate the overall min and max from the collected normalized and rounded values
norm_min_x = np.round(np.min(norm_min_x, axis=0), 4)
norm_max_x = np.round(np.max(norm_max_x, axis=0), 4)
norm_min_edge_attr = np.round(np.min(norm_min_edge_attr, axis=0), 4)
norm_max_edge_attr = np.round(np.max(norm_max_edge_attr, axis=0), 4)

# Prepare the final DataFrame with the global rounded min and max values
norm_rounded_stats_df = pd.DataFrame({
    'feature': feature_names_x + feature_names_edge_attr,
    'rounded_min': np.concatenate((norm_min_x, norm_min_edge_attr)),
    'rounded_max': np.concatenate((norm_max_x, norm_max_edge_attr))
})

norm_rounded_stats_df.to_csv(norm_stats, index=False)
