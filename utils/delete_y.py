# Delete GO Labels to replace with ESM2 labels for pretraining after assigning GO stratified splits
import os
import torch
from torch_geometric.data import Data

def delete_y(directory):
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            file_path = os.path.join(directory, filename)
            data = torch.load(file_path)
            if hasattr(data, 'y'):
                delattr(data, 'y')
            torch.save(data, file_path)

if __name__ == "__main__":
    directory = '/content/41k_sample_processed_GO'
    delete_y(directory)
