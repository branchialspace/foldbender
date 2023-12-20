# Ensure mirrored edges
import os
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected
from tqdm import tqdm

def mirror_edges(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]
    for filename in tqdm(files, desc="Processing files"):
        file_path = os.path.join(directory, filename)
        data = torch.load(file_path)

        # Ensure the data object has edge_index and edge_attr
        if hasattr(data, 'edge_index') and hasattr(data, 'edge_attr'):
            # Convert to undirected graph
            data.edge_index, data.edge_attr = to_undirected(data.edge_index, data.edge_attr)
            
            # Save the processed data object
            torch.save(data, file_path)
        else:
            print(f"Skipped {filename} as it does not contain necessary attributes")

mirror_edges('/content/41k_final_esm2')
