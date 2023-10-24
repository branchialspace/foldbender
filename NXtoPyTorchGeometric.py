# NetworkX > PyTorch Geometric Graph Representation of proteins
# Currently reads and writes to GCS - TODO: Abstract reading and writing files away.
import os
import torch
import networkx as nx
from sklearn import preprocessing
from collections import defaultdict
import pickle
from google.cloud import storage
import tempfile
from torch_geometric.data import Data
import numpy as np
from google.cloud import storage

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/content/drive/MyDrive/instant-tape-*****.json"
storage_client = storage.Client()

input_bucket = storage_client.get_bucket('proteins_sample')
blobs = list(input_bucket.list_blobs())
output_bucket = storage_client.get_bucket('pyg-molecular')

# Initialize OneHotEncoders for atom_name, atom_type, residue_name and secondary_structure
ohe_atom_names = preprocessing.OneHotEncoder(sparse_output=False)
ohe_atom_types = preprocessing.OneHotEncoder(sparse_output=False)
ohe_residue_names = preprocessing.OneHotEncoder(sparse_output=False)
ohe_secondary_structures = preprocessing.OneHotEncoder(sparse_output=False)

# Collect unique categorical values for each feature
unique_atom_names = set()
unique_atom_types = set()
unique_residue_names = set()
unique_secondary_structures = set()

# Iterate over all blobs
for blob in blobs:
    if blob.name.endswith(".pickle"):
        # Create a temporary file to download the blob content
        with tempfile.NamedTemporaryFile() as temp_file:
            blob.download_to_file(temp_file)
            temp_file.seek(0)

            # Load the NetworkX graph
            with open(temp_file.name, 'rb') as file:
                G = pickle.load(file)

            for node, data in G.nodes(data=True):
                unique_atom_names.add(data['atom_name'])
                unique_atom_types.add(data['atomic_number'])
                unique_residue_names.add(data['residue_name'])
                unique_secondary_structures.add(data['secondary_structure'])

# Fit the OneHotEncoders
ohe_atom_names.fit(np.array(list(unique_atom_names)).reshape(-1, 1))
ohe_atom_types.fit(np.array(list(unique_atom_types)).reshape(-1, 1))
ohe_residue_names.fit(np.array(list(unique_residue_names)).reshape(-1, 1))
ohe_secondary_structures.fit(np.array(list(unique_secondary_structures)).reshape(-1, 1))

# Iterate over the blobs again to create and save PyG graphs
for blob in blobs:
    if blob.name.endswith(".pickle"):
        # Create a temporary file to download the blob content
        with tempfile.NamedTemporaryFile() as temp_file:
            blob.download_to_file(temp_file)
            temp_file.seek(0)

            # Load the NetworkX graph
            with open(temp_file.name, 'rb') as file:
                G = pickle.load(file)

            # Add mirrored edges if they do not exist
            for node1, node2, data in G.edges(data=True):
                if not G.has_edge(node2, node1):
                    G.add_edge(node2, node1, **data)

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
                                      data['O_NH_2_energy']]], dtype=torch.float)
                    ], dim=0))


            for node1, node2, data in G.edges(data=True):
                # edge feature includes bond_idx, bond_order, bond_length and pae
                if 'pae' in data:
                    edge_feat.append([0, 0, 0, data.get('pae')])
                else:
                    edge_feat.append([
                        data.get('bond_idx', 0),
                        data.get('bond_order', 0),
                        data.get('bond_length', 0),
                        0])  # 0 for PAE in bond edges
                edge_index.append((node_mapping[node1], node_mapping[node2]))

            # Convert lists to tensors
            edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
            feat = torch.stack(feat)
            edge_feat = torch.tensor(edge_feat, dtype=torch.float)
            atom_coords = torch.stack(atom_coords_list)

            # Construct the PyG graph
            data = Data(edge_index=edge_index, edge_weight=edge_weight, num_nodes=len(G), feat=feat, edge_feat=edge_feat, atom_coords=atom_coords)

            # Construct the dictionary and save it using PyTorch's serialization
            data_dict = {'graph_with_features': data}
            output_filename = blob.name.replace('.pickle', '.pkl') # Change extension to .pt
            torch.save(data_dict, output_filename)
            # Save to the output bucket
            output_blob = output_bucket.blob(output_filename)
            output_blob.upload_from_filename(output_filename)
