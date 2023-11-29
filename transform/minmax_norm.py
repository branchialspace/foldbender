# minmax normalization
import os
import torch
import numpy as np
import pandas as pd
import zipfile
import tempfile
from pathlib import Path

def minmax_norm(input_dir, output_dir, zip_io=True):
    os.makedirs(output_dir, exist_ok=True)
    stats = os.path.join(os.path.dirname(output_dir), f"{os.path.basename(output_dir)}_stats.csv")
    norm_stats = os.path.join(os.path.dirname(output_dir), f"{os.path.basename(output_dir)}_norm_stats.csv")

    # Initialize global min and max arrays
    global_x_min, global_x_max = None, None
    global_edge_attr_min, global_edge_attr_max = None, None

    def process_file(file_path):
        nonlocal global_x_min, global_x_max, global_edge_attr_min, global_edge_attr_max
        data_object = torch.load(file_path)

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

    # Process files (first pass)
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pt') or (zip_io and file_name.endswith('.zip')):
            data_path = os.path.join(input_dir, file_name)
            
            if zip_io and file_name.endswith('.zip'):
                with zipfile.ZipFile(data_path, 'r') as zip_ref:
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        zip_ref.extractall(tmpdirname)
                        for extracted_file in os.listdir(tmpdirname):
                            process_file(os.path.join(tmpdirname, extracted_file))
            else:
                process_file(data_path)

    # Save global min/max stats
    feature_names_x = [f'x_feature_{i}' for i in range(len(global_x_min))]
    feature_names_edge_attr = [f'edge_attr_feature_{i}' for i in range(len(global_edge_attr_min))]
    scaling_parameters_df = pd.DataFrame({
        'feature': feature_names_x + feature_names_edge_attr,
        'min': np.concatenate((global_x_min, global_edge_attr_min)),
        'max': np.concatenate((global_x_max, global_edge_attr_max))
    })
    scaling_parameters_df.to_csv(stats, index=False)

    accumulated_x_mins, accumulated_x_maxs = [], []
    accumulated_edge_attr_mins, accumulated_edge_attr_maxs = [], []

    # Function to scale and save data objects
    def scale_and_save(file_path, output_path):
        nonlocal accumulated_x_mins, accumulated_x_maxs, accumulated_edge_attr_mins, accumulated_edge_attr_maxs
        data_object = torch.load(file_path)

        # Scale data
        data_object.x = (data_object.x - torch.tensor(global_x_min, dtype=torch.float32)) / \
                        (torch.tensor(global_x_max, dtype=torch.float32) - torch.tensor(global_x_min, dtype=torch.float32))
        data_object.edge_attr = (data_object.edge_attr - torch.tensor(global_edge_attr_min, dtype=torch.float32)) / \
                                (torch.tensor(global_edge_attr_max, dtype=torch.float32) - torch.tensor(global_edge_attr_min, dtype=torch.float32))

        data_object.x[torch.isnan(data_object.x)] = 0.0
        data_object.x = torch.round(data_object.x * 1000000) / 1000000
        data_object.edge_attr = torch.round(data_object.edge_attr * 1000000) / 1000000

        accumulated_x_mins.append(data_object.x.min(dim=0).values)
        accumulated_x_maxs.append(data_object.x.max(dim=0).values)
        accumulated_edge_attr_mins.append(data_object.edge_attr.min(dim=0).values)
        accumulated_edge_attr_maxs.append(data_object.edge_attr.max(dim=0).values)

        # Save data object
        if zip_io:
            with tempfile.TemporaryDirectory() as tmpdirname:
                temp_file_path = os.path.join(tmpdirname, 'temp.pt')
                torch.save(data_object, temp_file_path)
                with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                    zipf.write(temp_file_path, Path(temp_file_path).name)
        else:
            torch.save(data_object, output_path)

    # Second pass to scale the data using the global min and max
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.pt') or (zip_io and file_name.endswith('.zip')):
            data_path = os.path.join(input_dir, file_name)
            output_file_name = file_name.replace('.pt', '.zip') if zip_io else file_name
            output_path = os.path.join(output_dir, output_file_name)

            if zip_io and file_name.endswith('.zip'):
                with zipfile.ZipFile(data_path, 'r') as zip_ref:
                    with tempfile.TemporaryDirectory() as tmpdirname:
                        zip_ref.extractall(tmpdirname)
                        for extracted_file in os.listdir(tmpdirname):
                            scale_and_save(os.path.join(tmpdirname, extracted_file), output_path)
            else:
                scale_and_save(data_path, output_path)

    # Save normalized rounded stats
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
    output_dir = '/content/drive/MyDrive/protein-DATA/sample-normalized'
    
    minmax_norm(input_dir, output_dir)
