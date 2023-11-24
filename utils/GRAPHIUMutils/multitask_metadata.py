# Create metadata files
import os
import torch
from torch_geometric.data import Data

def extract_metadata(graph_data_list):
    # Initialize the metadata dictionary
    metadata = {
        'mol_ids': None,
        'smiles': None,
        'labels_size': {},
        'labels_dtype': {},
        'dataset_length': len(graph_data_list),
        '_num_nodes_list': [],
        '_num_edges_list': []
    }

    # Iterate over graph_data_list and populate metadata dictionary
    for data_obj in graph_data_list:
        metadata['_num_nodes_list'].append(data_obj['graph_with_features'].num_nodes)
        metadata['_num_edges_list'].append(data_obj['graph_with_features'].edge_feat.size(0))  # Append the number of edges

        for key, value in data_obj['labels']:
            if key.startswith("GO:"):
                metadata['labels_size'][key] = value.size()
                metadata['labels_dtype'][key] = value.dtype

    return metadata

# Define the parent directory and subdirectories
parent_dir = "/datacache/"
subdirs = ["train", "val", "test"]

# Iterate over subdirectories
for subdir in subdirs:
    # Initialize a list to store the PyTorch Geometric data objects
    graph_data_list = []

    # Directory path
    dir_path = os.path.join(parent_dir, subdir)

    # Iterate over .pkl files in the subdirectory
    for filename in os.listdir(dir_path):
        if filename.endswith(".pkl"):
            file_path = os.path.join(dir_path, filename)

            # Load the PyTorch Geometric data object using torch.load
            graph_data = torch.load(file_path)
            graph_data_list.append(graph_data)

    # Extract metadata from the graph_data_list
    metadata = extract_metadata(graph_data_list)

    # Save the metadata using torch.save in the respective subdirectory
    torch.save(metadata, os.path.join(dir_path, "multitask_metadata.pkl"))
