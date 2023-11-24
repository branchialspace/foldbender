# minmax normalization
import os
import torch
import numpy as np
import pandas as pd

def minmax_norm(input_dir, output_dir, stats, norm_stats):
    os.makedirs(output_directory, exist_ok=True)
    
    # Initialize global min and max arrays
    global_x_min = None
    global_x_max = None
    global_edge_attr_min = None
    global_edge_attr_max = None

    # First pass to find the global min and max for each feature
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pt'):
            data_path = os.path.join(input_dir, file_name)
            data_object = torch.load(data_path)

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
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pt'):
            data_path = os.path.join(input_dir, file_name)
            data_object = torch.load(data_path)

            data_object.x = (data_object.x - torch.tensor(global_x_min, dtype=torch.float32)) / \
                            (torch.tensor(global_x_max, dtype=torch.float32) - torch.tensor(global_x_min, dtype=torch.float32))
            data_object.edge_attr = (data_object.edge_attr - torch.tensor(global_edge_attr_min, dtype=torch.float32)) / \
                                    (torch.tensor(global_edge_attr_max, dtype=torch.float32) - torch.tensor(global_edge_attr_min, dtype=torch.float32))

            data_object.x[torch.isnan(data_object.x)] = 0.0
            data_object.x = torch.round(data_object.x * 10000) / 10000
            data_object.edge_attr = torch.round(data_object.edge_attr * 10000) / 10000
            data_object.atom_coords = torch.round(data_object.atom_coords * 10000) / 10000

            accumulated_x_mins.append(data_object.x.min(dim=0).values)
            accumulated_x_maxs.append(data_object.x.max(dim=0).values)
            accumulated_edge_attr_mins.append(data_object.edge_attr.min(dim=0).values)
            accumulated_edge_attr_maxs.append(data_object.edge_attr.max(dim=0).values)
            
            output_path = os.path.join(output_dir, file_name)
            torch.save(data_object, output_path)

    global_rounded_x_min = torch.stack(accumulated_x_mins).min(dim=0).values.numpy()
    global_rounded_x_max = torch.stack(accumulated_x_maxs).max(dim=0).values.numpy()
    global_rounded_edge_attr_min = torch.stack(accumulated_edge_attr_mins).min(dim=0).values.numpy()
    global_rounded_edge_attr_max = torch.stack(accumulated_edge_attr_maxs).max(dim=0).values.numpy()

    norm_rounded_stats_df = pd.DataFrame({
        'feature': feature_names_x + feature_names_edge_attr,
        'rounded_min': np.round(np.concatenate((global_rounded_x_min, global_rounded_edge_attr_min)), 4),
        'rounded_max': np.round(np.concatenate((global_rounded_x_max, global_rounded_edge_attr_max)), 4)
    })
    norm_rounded_stats_df.to_csv(norm_stats, index=False)

if __name__ == "__main__":

    input_dir = '/content/drive/MyDrive/protein-DATA/prot-sample'
    output_dir = '/content/drive/MyDrive/protein-DATA/sample-normalized'
    stats = '/content/drive/MyDrive/protein-DATA/minmax-stats.csv'
    norm_stats = '/content/drive/MyDrive/protein-DATA/norm-stats.csv'
    
    minmax_norm(input_dir, output_dir, stats, norm_stats)
