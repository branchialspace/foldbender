# Deep Graph Library Graph Representation of Proteins
# Currently takes input from GraphML (excpects GCS blob structure of directory-subdrectory-GraphML)
# TODO: Change input to NX 
import os
import dgl
from dgl import DGLGraph
from dgl import heterograph
import torch
import networkx as nx
from sklearn import preprocessing
from collections import defaultdict

# a dictionary to map node labels to integers
node_mapping = defaultdict(int)

# Initialize LabelEncoders for atom_name, atom_type, residue_name and secondary_structure
le_atom_names = preprocessing.LabelEncoder()
le_atom_types = preprocessing.LabelEncoder()
le_residue_names = preprocessing.LabelEncoder()
le_secondary_structures = preprocessing.LabelEncoder()

# Collect unique categorical values for each feature
unique_atom_names = set()
unique_atom_types = set()
unique_residue_names = set()
unique_secondary_structures = set()

# Loop through all subdirectories in the parent directory
for directory_name in os.listdir(INPUT_DATA):
    directory_path = os.path.join(INPUT_DATA, directory_name)
    if os.path.isdir(directory_path):
        # Loop through all files in the subdirectory
        for filename in os.listdir(directory_path):
            if filename.endswith(".graphml"):
                file_path = os.path.join(directory_path, filename)

                # Load the NetworkX graph
                G = nx.read_graphml(file_path)

                for node, data in G.nodes(data=True):
                    unique_atom_names.add(data['atom_name'])
                    unique_atom_types.add(data['atomic_number'])
                    unique_residue_names.add(data['residue_name'])
                    unique_secondary_structures.add(data['secondary_structure'])

# Fit the LabelEncoders
le_atom_names.fit(list(unique_atom_names))
le_atom_types.fit(list(unique_atom_types))
le_residue_names.fit(list(unique_residue_names))
le_secondary_structures.fit(list(unique_secondary_structures))

# Iterate over the directories again to create and save DGL graphs
for directory_name in os.listdir(INPUT_DATA):
    directory_path = os.path.join(INPUT_DATA, directory_name)
    if os.path.isdir(directory_path):
        # Loop through all files in the subdirectory
        for filename in os.listdir(directory_path):
            if filename.endswith(".graphml"):
                file_path = os.path.join(directory_path, filename)

                # Load the NetworkX graph
                G = nx.read_graphml(file_path)

                # Map node labels to integers
                for i, node in enumerate(G.nodes()):
                    node_mapping[node] = i

                # Prepare features for edges
                edge_type1_features = []
                edge_type2_features = []
                edge_list_type1 = []
                edge_list_type2 = []

                # Create separate lists for each attribute
                atom_names = []
                atom_types = []
                residue_names = []
                atom_coords = []
                degrees = []
                aromatics = []
                residue_numbers = []
                plddts = []
                secondary_structures = []
                exposures = []
                phis = []
                psis = []
                NH_O_1_relidxs = []
                NH_O_1_energies = []
                O_NH_1_relidxs = []
                O_NH_1_energies = []
                NH_O_2_relidxs = []
                NH_O_2_energies = []
                O_NH_2_relidxs = []
                O_NH_2_energies = []
                paes = []
                bond_idxs = []
                bond_orders = []
                bond_lengths = []

                for node, data in G.nodes(data=True):
                    # Convert categorical variables using the label encoders
                    atom_names.append(le_atom_names.transform([data['atom_name']])[0])
                    atom_types.append(le_atom_types.transform([data['atomic_number']])[0])
                    residue_names.append(le_residue_names.transform([data['residue_name']])[0])
                    secondary_structures.append(le_secondary_structures.transform([data['secondary_structure']])[0])

                    # Convert string of atom coordinates to tensor
                    atom_coords.append(torch.tensor([float(coord) for coord in data['atom_coords'].split(',')], dtype=torch.float32))

                    # Add additional node features
                    degrees.append(data['degree'])
                    aromatics.append(data['aromatic'])
                    residue_numbers.append(data['residue_number'])
                    plddts.append(data['plddt'])
                    exposures.append(data['exposure'])
                    phis.append(data['phi'])
                    psis.append(data['psi'])
                    NH_O_1_relidxs.append(data['NH_O_1_relidx'])
                    NH_O_1_energies.append(data['NH_O_1_energy'])
                    O_NH_1_relidxs.append(data['O_NH_1_relidx'])
                    O_NH_1_energies.append(data['O_NH_1_energy'])
                    NH_O_2_relidxs.append(data['NH_O_2_relidx'])
                    NH_O_2_energies.append(data['NH_O_2_energy'])
                    O_NH_2_relidxs.append(data['O_NH_2_relidx'])
                    O_NH_2_energies.append(data['O_NH_2_energy'])

                for node1, node2, data in G.edges(data=True):
                    # Add additional edge features
                    if 'pae' in data:
                        edge_list_type2.append((node_mapping[node1], node_mapping[node2]))
                        paes.append(data['pae'])
                    else:
                        edge_list_type1.append((node_mapping[node1], node_mapping[node2]))
                        bond_idxs.append(data['bond_idx'])
                        bond_orders.append(data['bond_order'])
                        bond_lengths.append(data['bond_length'])

                # Construct the heterogeneous graph
                data_dict = {
                    ('node', 'edge_type1', 'node'): (torch.tensor([edge[0] for edge in edge_list_type1]), torch.tensor([edge[1] for edge in edge_list_type1])),
                    ('node', 'edge_type2', 'node'): (torch.tensor([edge[0] for edge in edge_list_type2]), torch.tensor([edge[1] for edge in edge_list_type2])),
                }

                g = heterograph(data_dict)

                # Add node features into the DGL Graph
                g.nodes['node'].data['atom_name'] = torch.tensor(atom_names)
                g.nodes['node'].data['atomic_number'] = torch.tensor(atom_types)
                g.nodes['node'].data['residue_name'] = torch.tensor(residue_names)
                g.nodes['node'].data['atom_coords'] = torch.stack(atom_coords)
                g.nodes['node'].data['degree'] = torch.tensor(degrees)
                g.nodes['node'].data['aromatic'] = torch.tensor(aromatics)
                g.nodes['node'].data['residue_number'] = torch.tensor(residue_numbers)
                g.nodes['node'].data['plddt'] = torch.tensor(plddts)
                g.nodes['node'].data['secondary_structure'] = torch.tensor(secondary_structures)
                g.nodes['node'].data['exposure'] = torch.tensor(exposures)
                g.nodes['node'].data['phi'] = torch.tensor(phis)
                g.nodes['node'].data['psi'] = torch.tensor(psis)
                g.nodes['node'].data['NH_O_1_relidx'] = torch.tensor(NH_O_1_relidxs)
                g.nodes['node'].data['NH_O_1_energy'] = torch.tensor(NH_O_1_energies)
                g.nodes['node'].data['O_NH_1_relidx'] = torch.tensor(O_NH_1_relidxs)
                g.nodes['node'].data['O_NH_1_energy'] = torch.tensor(O_NH_1_energies)
                g.nodes['node'].data['NH_O_2_relidx'] = torch.tensor(NH_O_2_relidxs)
                g.nodes['node'].data['NH_O_2_energy'] = torch.tensor(NH_O_2_energies)
                g.nodes['node'].data['O_NH_2_relidx'] = torch.tensor(O_NH_2_relidxs)
                g.nodes['node'].data['O_NH_2_energy'] = torch.tensor(O_NH_2_energies)

                # Add edge features into the DGL Graph
                g.edges['edge_type1'].data['bond_idx'] = torch.tensor(bond_idxs)
                g.edges['edge_type1'].data['bond_order'] = torch.tensor(bond_orders)
                g.edges['edge_type1'].data['bond_length'] = torch.tensor(bond_lengths)
                g.edges['edge_type2'].data['pae'] = torch.tensor(paes)

                # Save the DGL Graph
                dgl_file_path = os.path.join(directory_path, filename.replace('.graphml', '.dgl'))
                dgl.save_graphs(dgl_file_path, g)
