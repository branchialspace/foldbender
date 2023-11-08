# Atomic Positional Encodings
import torch
from dscribe.descriptors import SOAP
from torch_geometric.data import Data
import os

input_directory = '/content/drive/MyDrive/protein-DATA/sample-normalized'
output_directory = '/content/drive/MyDrive/protein-DATA/sample-atomic-encoded'

def calculate_bond_distances(edge_index, edge_attr, num_atoms, atom_coords):
    # Create a dictionary to store bond distances for each atom
    bond_distances = {i: [] for i in range(num_atoms)}

    # Helper function to calculate Euclidean distance between two atoms
    def euclidean_distance(idx1, idx2):
        coord1, coord2 = atom_coords[idx1], atom_coords[idx2]
        return torch.sqrt(torch.sum((coord1 - coord2) ** 2))

    # Calculate distances for atoms that are one bond away
    for bond in range(edge_index.shape[1]):
        if edge_attr[bond][0] != 0:  # If there's a bond
            atom_i = edge_index[0, bond].item()
            atom_j = edge_index[1, bond].item()
            distance = euclidean_distance(atom_i, atom_j)
            bond_distances[atom_i].append((atom_j, distance.item()))
            bond_distances[atom_j].append((atom_i, distance.item()))  # Add the reverse as well

    # Calculate distances for atoms that are two and three bonds away
    for atom in range(num_atoms):
        # Get one-bond neighbors
        one_bond_neighbors = [neighbor for neighbor, _ in bond_distances[atom]]
        # Iterate over one-bond neighbors to find two and three-bond neighbors
        for neighbor in one_bond_neighbors:
            # Get one-bond neighbors of the one-bond neighbor (two-bond neighbors of the atom)
            two_bond_neighbors = [second_neighbor for second_neighbor, _ in bond_distances[neighbor] if second_neighbor != atom]
            for second_neighbor in two_bond_neighbors:
                # Calculate the distance from the atom to the two-bond neighbor
                distance = euclidean_distance(atom, second_neighbor)
                bond_distances[atom].append((second_neighbor, distance.item()))
                # Get one-bond neighbors of the two-bond neighbor (three-bond neighbors of the atom)
                three_bond_neighbors = [third_neighbor for third_neighbor, _ in bond_distances[second_neighbor] if third_neighbor != neighbor and third_neighbor != atom]
                for third_neighbor in three_bond_neighbors:
                    # Calculate the distance from the atom to the three-bond neighbor
                    distance = euclidean_distance(atom, third_neighbor)
                    bond_distances[atom].append((third_neighbor, distance.item()))

    # For each atom, keep only the unique atom-distance pairs with the maximum distance
    for atom in bond_distances:
        unique_atoms = set()
        unique_bond_distances = []
        for atom_id, dist in bond_distances[atom]:
            if atom_id not in unique_atoms:
                unique_atoms.add(atom_id)
                unique_bond_distances.append((atom_id, dist))
        # Replace the list with the list of unique pairs
        bond_distances[atom] = unique_bond_distances

    # Sort the lists by distance and keep the farthest one
    for atom in bond_distances:
        bond_distances[atom].sort(key=lambda x: x[1], reverse=True)

    return bond_distances

def soap_local(input_directory, output_directory):
    # Initialize the SOAP descriptor
    soap = SOAP(species=["x", "y"], periodic=False, n_max=3, l_max=3, sigma=0.1)
    
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

            # Calculate bond distances for each atom using atom coordinates
            bond_distances = calculate_bond_distances(edge_index, edge_attr, num_atoms, atom_coords)

            # This will hold the local SOAP descriptors
            local_descriptors = []

            # Calculate SOAP descriptor for each atom
            for i in range(num_atoms):
                # Define the current atom as "x" and all others as "y"
                species = ["y"] * num_atoms
                species[i] = "x"  # The current atom is of type "x"

                # Update rcut for SOAP descriptor to be the distance to the farthest 3-bond distance atom
                farthest_distance = bond_distances[i][0][1]  # Get the maximum distance
                soap.set_rcut(farthest_distance + 1e-3)  # Add a small buffer to include the farthest atom

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
