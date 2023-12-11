# Delete GO Labels to replace with ESM2 labels for pretraining after assigning GO stratified splits
import os
import torch
from torch_geometric.data import Data

def delete_y(input_dir, output_dir):
    os.makedirs(output_directory, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.pt'):
            file_path = os.path.join(input_dir, filename)
            data = torch.load(file_path)
            if hasattr(data, 'y'):
                delattr(data, 'y')
            output_file_path = os.path.join(output_dir, filename)
            torch.save(data, output_file_path)

if __name__ == "__main__":

    input_directory = '/content/drive/MyDrive/protein-DATA/41k_sample_processed_GO'
    output_directory = '/content/drive/MyDrive/protein-DATA/41k_sample_processed_noY'
    process_pyg_data(input_directory, output_directory)
