# minmax normalization
import os
import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.data import Data

input_dir = '/content/drive/MyDrive/protein-DATA/prot-pyg-sample'
output_dir = '/content/drive/MyDrive/protein-DATA/sample-normalized'
stats = '/content/drive/MyDrive/protein-DATA/minmax-stats.csv'

# Initialize global min and max arrays
global_x_min = None
global_x_max = None
global_edge_attr_min = None
global_edge_attr_max = None

# First pass to find the global min and max for each feature
for file_name in os.listdir(input_dir):
    if file_name.endswith('.pt'):
        # Load the data object
        data_path = os.path.join(input_dir, file_name)
        data_dict = torch.load(data_path)

        # Extract the key that matches the file name without the '.pt' extension
        data_key = file_name[:-3]
        data_object = data_dict[data_key]

        # If it's the first file, initialize the global min and max
        if global_x_min is None:
            global_x_min = data_object.x.min(dim=0).values.cpu().numpy()
            global_x_max = data_object.x.max(dim=0).values.cpu().numpy()
            global_edge_attr_min = data_object.edge_attr.min(dim=0).values.cpu().numpy()
            global_edge_attr_max = data_object.edge_attr.max(dim=0).values.cpu().numpy()
        else:
            # Update global min and max for each feature in x
            global_x_min = np.minimum(global_x_min, data_object.x.min(dim=0).values.cpu().numpy())
            global_x_max = np.maximum(global_x_max, data_object.x.max(dim=0).values.cpu().numpy())

            # Update global min and max for each feature in edge_attr
            global_edge_attr_min = np.minimum(global_edge_attr_min, data_object.edge_attr.min(dim=0).values.cpu().numpy())
            global_edge_attr_max = np.maximum(global_edge_attr_max, data_object.edge_attr.max(dim=0).values.cpu().numpy())

# Create a DataFrame for storing feature-wise min and max
feature_names_x = [f'x_feature_{i}' for i in range(len(global_x_min))]
feature_names_edge_attr = [f'edge_attr_feature_{i}' for i in range(len(global_edge_attr_min))]

scaling_parameters_df = pd.DataFrame({
    'feature': feature_names_x + feature_names_edge_attr,
    'min': np.concatenate((global_x_min, global_edge_attr_min)),
    'max': np.concatenate((global_x_max, global_edge_attr_max))
})

# Second pass to scale the data using the global min and max
for file_name in os.listdir(input_dir):
    if file_name.endswith('.pt'):
        # Load the data object
        data_path = os.path.join(input_dir, file_name)
        data_dict = torch.load(data_path)
        
        # Extract the key that matches the file name without the '.pt' extension
        data_key = file_name[:-3]
        data_object = data_dict[data_key]

        # Normalize each feature using the global min and max values
        data_object.x = (data_object.x - torch.tensor(global_x_min, dtype=torch.float32)) / \
                        (torch.tensor(global_x_max, dtype=torch.float32) - torch.tensor(global_x_min, dtype=torch.float32))
        data_object.edge_attr = (data_object.edge_attr - torch.tensor(global_edge_attr_min, dtype=torch.float32)) / \
                                (torch.tensor(global_edge_attr_max, dtype=torch.float32) - torch.tensor(global_edge_attr_min, dtype=torch.float32))

        # Update the dictionary with the scaled Data object
        data_dict[data_key] = data_object

        # Save the modified data dictionary
        output_path = os.path.join(output_dir, file_name)
        torch.save(data_dict, output_path)

# Save the scaling parameters to a CSV file
scaling_parameters_df.to_csv(stats, index=False)
