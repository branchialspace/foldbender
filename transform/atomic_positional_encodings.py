import torch
from dscribe.descriptors import SOAP
from torch_geometric.data import Data
import os

input_directory = '/content/drive/MyDrive/protein-DATA/sample-normalized'
output_directory = '/content/drive/MyDrive/protein-DATA/sample-atomic-encoded'

def calculate_bond_distances(edge_index, edge_attr, num_atoms):
    # Create a dictionary to store bond distances for each atom
    bond_distances = {}
    
    for i in range(num_atoms):
        # Find all edges connected to the current atom
        connected_edges = edge_index[:, edge_index[0] == i]
        
        # Extract bond distances where the first position in edge_attr is not 0 (indicating a bond)
        bond_distances[i] = [edge_attr[j] for j in range(connected_edges.size(1)) if edge_attr[j][0] != 0]
    
    return bond_distances

def soap_local(input_directory, output_directory):
    # Initialize the SOAP descriptor
    soap = SOAP(species=["x", "y"], periodic=False, rcut=3.0, nmax=3, lmax=3, sigma=0.1)
    
    # Create the output directory if it does not exist
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Iterate over the .pt files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.pt'):
            # Load the dictionary containing the PyTorch Geometric Data object
            file_path = os.path.join(input_directory, filename)
            data_dict = torch.load(file_path)

            # Extract the key for the data
            data_key = filename.rstrip('.pt')
            data = data_dict[data_key]

            # Retrieve the atom coordinates and edge information
            atom_coords = data['atom_coords']
            edge_index = data['edge_index']
            edge_attr = data['edge_attr']
            num_atoms = len(atom_coords)

            # Calculate bond distances for each atom
            bond_distances = calculate_bond_distances(edge_index, edge_attr, num_atoms)

            # This will hold the local SOAP descriptors
            local_descriptors = []

            # Calculate SOAP descriptor for each atom
            for i in range(num_atoms):
                # Define the current atom as "x" and all others as "y"
                species = ["y"] * num_atoms
                species[i] = "x"  # The current atom is of type "x"

                # Determine the farthest 3-bond distance atom for the current atom
                farthest_atom = max(bond_distances[i], key=lambda x: x[1])

                # Update rcut for SOAP descriptor to be the distance to the farthest 3-bond distance atom
                soap.set_rcut(farthest_atom[1])

                # Calculate the descriptor for the current atom
                descriptor = soap.create(atom_coords, species=species)
                local_descriptors.append(descriptor[i])  # Only append the descriptor for the central atom

            # Add the local descriptors to the data object
            data['local_soap'] = torch.tensor(local_descriptors, dtype=torch.float32)  # Ensure correct tensor type

            # Update the dictionary with the modified data object
            data_dict[data_key] = data

            # Define the path for the output file
            output_file_path = os.path.join(output_directory, filename)

            # Save the updated dictionary back to disk in the separate directory
            torch.save(data_dict, output_file_path)
            
    print(f"All local SOAP descriptors calculated and saved to {output_directory}.")

soap_local(input_directory, output_directory)
