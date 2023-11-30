# NetworkX > PyTorch Geometric Graph Representation of proteins
import os
import torch
from multiprocessing import Pool, cpu_count
import networkx as nx
from sklearn import preprocessing
from collections import defaultdict
import pickle
from torch_geometric.data import Data
import numpy as np
import csv

def save_encoders(encoders, directory):
    for name, encoder in encoders.items():
        with open(os.path.join(directory, f'{name}_encoder.pkl'), 'wb') as f:
            pickle.dump(encoder, f)

def load_encoders(directory):
    encoders = {}
    for name in ['atom_names', 'atom_types', 'residue_names', 'secondary_structures']:
        with open(os.path.join(directory, f'{name}_encoder.pkl'), 'rb') as f:
            encoders[name] = pickle.load(f)
    return encoders

def process_categories(input_dir, categories_path):
    # Initialize encoders
    ohe_atom_names, ohe_atom_types, ohe_residue_names, ohe_secondary_structures = (
        preprocessing.OneHotEncoder(sparse_output=False),
        preprocessing.OneHotEncoder(sparse_output=False),
        preprocessing.OneHotEncoder(sparse_output=False),
        preprocessing.OneHotEncoder(sparse_output=False)
    )

    # Collect unique categories
    unique_atom_names, unique_atom_types, unique_residue_names, unique_secondary_structures = set(), set(), set(), set()
    for filename in os.listdir(input_dir):
        if filename.endswith(".pkl"):
            with open(os.path.join(input_dir, filename), 'rb') as file:
                G = pickle.load(file)
                for _, data in G.nodes(data=True):
                    unique_atom_names.add(data['atom_name'])
                    unique_atom_types.add(data['atomic_number'])
                    unique_residue_names.add(data['residue_name'])
                    unique_secondary_structures.add(data['secondary_structure'])
    
    for ohe, unique_values in zip([ohe_atom_names, ohe_atom_types, ohe_residue_names, ohe_secondary_structures],
                                  [unique_atom_names, unique_atom_types, unique_residue_names, unique_secondary_structures]):
        ohe.fit(np.array(list(unique_values)).reshape(-1, 1))

    # Write categories to file
    with open(categories_path, 'w', newline='') as csvfile:
        category_writer = csv.writer(csvfile)
        for category_name, unique_values in zip(['atom_name', 'atom_type', 'residue_name', 'secondary_structure'], 
                                                [unique_atom_names, unique_atom_types, unique_residue_names, unique_secondary_structures]):
            category_writer.writerow([category_name] + list(unique_values))

    # Save the encoders
    encoders = {
        'atom_names': ohe_atom_names,
        'atom_types': ohe_atom_types,
        'residue_names': ohe_residue_names,
        'secondary_structures': ohe_secondary_structures
    }
    save_encoders(encoders, os.path.dirname(categories_path))

    return ohe_atom_names, ohe_atom_types, ohe_residue_names, ohe_secondary_structures

def process_graph(filename, input_dir, output_dir, encoders, include_pae=False):
    ohe_atom_names, ohe_atom_types, ohe_residue_names, ohe_secondary_structures = encoders
    data_object_name = filename.replace('.pkl', '')
    filepath = os.path.join(input_dir, filename)

    # Load the NetworkX graph
    with open(filepath, 'rb') as file:
        G = pickle.load(file)

        # a dictionary to map node IDs to integers
        node_mapping = defaultdict(int)

        # Map node labels to integers
        for i, node in enumerate(G.nodes()):
            node_mapping[node] = i

        # Prepare features for edges
        edge_index = []
        edge_feat = []

        # Prepare node features
        feat = []

        # Prepare atom coordinates separately
        atom_coords_list = []

        for node, data in G.nodes(data=True):
            # Node features
            atom_coords = torch.tensor([float(i) for i in data['atom_coords'].split(",")])
            atom_coords_list.append(atom_coords)
            feat.append(torch.cat([torch.tensor(ohe_atom_names.transform([[data['atom_name']]])).squeeze(0),
                    torch.tensor(ohe_atom_types.transform([[data['atomic_number']]])).squeeze(0),
                    torch.tensor(ohe_residue_names.transform([[data['residue_name']]])).squeeze(0),
                    torch.tensor(ohe_secondary_structures.transform([[data['secondary_structure']]])).squeeze(0),
                    torch.tensor([[data['degree'],
                                  data['aromatic'],
                                  data['residue_number'],
                                  data['plddt'],
                                  data['exposure'],
                                  data['phi'],
                                  data['psi'],
                                  data['NH_O_1_relidx'],
                                  data['NH_O_1_energy'],
                                  data['O_NH_1_relidx'],
                                  data['O_NH_1_energy'],
                                  data['NH_O_2_relidx'],
                                  data['NH_O_2_energy'],
                                  data['O_NH_2_relidx'],
                                  data['O_NH_2_energy']]], dtype=torch.float).squeeze(0)
                ], dim=0))

        for node1, node2, data in G.edges(data=True):
            # Skip edges with 'pae' attribute if include_pae is False
            if not include_pae and 'pae' in data:
                continue
                
            # Edge feature construction
            if include_pae and 'pae' in data:
                edge_features = [
                    # data.get('bond_idx', 0),
                    data.get('bond_order', 0),
                    data.get('bond_length', 0),
                    data.get('pae', 0)
                ]
            else:
                edge_features = [
                    # data.get('bond_idx', 0),
                    data.get('bond_order', 0),
                    data.get('bond_length', 0)
                ]
            edge_feat.append(edge_features)
            edge_index.append((node_mapping[node1], node_mapping[node2]))
        
        # Convert lists to tensors
        edge_index = torch.LongTensor(edge_index).t().contiguous()
        feat = torch.stack(feat)
        edge_feat = torch.tensor(edge_feat, dtype=torch.float)

        # Calculate geometric center and re-align atom_coords
        atom_coords_array = torch.stack(atom_coords_list)
        geometric_center = torch.mean(atom_coords_array, dim=0)
        aligned_atom_coords_list = [coords - geometric_center for coords in atom_coords_array]

        # Update atom_coords with the aligned coords
        atom_coords = torch.stack(aligned_atom_coords_list)

        # Create the num_nodes attribute
        num_nodes = feat.shape[0]

        # Construct the PyG graph
        data = Data(edge_index=edge_index, x=feat, edge_attr=edge_feat, num_nodes=num_nodes, atom_coords=atom_coords)
    
        # Save the PyTorch object to the local file system
        output_filename = f'{data_object_name}.pt'
        torch.save(data, os.path.join(output_dir, output_filename))

def process_file(args):
    filename, input_dir, output_dir, categories_dir, include_pae = args

    # Load the encoders
    encoders = load_encoders(categories_dir)
    ohe_atom_names, ohe_atom_types, ohe_residue_names, ohe_secondary_structures = (
        encoders['atom_names'], encoders['atom_types'], encoders['residue_names'], encoders['secondary_structures']
    )

    process_graph(filename, input_dir, output_dir, (ohe_atom_names, ohe_atom_types, ohe_residue_names, ohe_secondary_structures), include_pae)

def nx_pyg(input_dir, output_dir, include_pae=False):
    os.makedirs(output_dir, exist_ok=True)
    categories_path = os.path.join(os.path.dirname(output_dir), f"{os.path.basename(os.path.normpath(output_dir))}_categories.csv")
    process_categories(input_dir, categories_path)

    files_to_process = [(filename, input_dir, output_dir, os.path.dirname(categories_path), include_pae) for filename in os.listdir(input_dir) if filename.endswith(".pkl")]

    num_processes = min(cpu_count(), len(files_to_process))

    with Pool(num_processes) as pool:
        pool.map(process_file, files_to_process)

if __name__ == "__main__":

    input_directory = '/proteins_sample/'
    output_directory = '/content/drive/MyDrive/protein-DATA/prot-sample/'
    
    nx_pyg(input_directory, output_directory)
