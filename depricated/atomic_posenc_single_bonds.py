# SOAP 3d-coordinate Positional Encodings
import torch
import numpy as np
import ase
from ase import Atoms
from dscribe.descriptors import SOAP
from torch_geometric.data import Data
import os

def soap_local(input_directory, output_directory):
    os.makedirs(output_directory, exist_ok=True)

    # Iterate over the .pt files in the directory
    for filename in os.listdir(input_directory):
        if filename.endswith('.pt'):
            # Load the PyTorch Geometric Data object
            file_path = os.path.join(input_directory, filename)
            data = torch.load(file_path)

            # Retrieve atom coordinates, edge indices, and existing 'x' tensor
            atom_coords = data['atom_coords']
            edge_index = data['edge_index']
            features_x = data['x']
            num_atoms = len(atom_coords)

            # Initialize SOAP descriptors list
            soap_descriptors_list = []

            # Iterate over each atom
            for i in range(num_atoms):
                # Find all atoms connected to atom i
                connected_atoms = [j for j in edge_index[1][edge_index[0] == i]] + [j for j in edge_index[0][edge_index[1] == i]]

                # Calculate distances to connected atoms and find the maximum
                max_distance = max((np.linalg.norm(atom_coords[i] - atom_coords[j]) for j in connected_atoms), default=0)

                # Set r_cut slightly beyond the maximum distance
                r_cut_dynamic = max_distance + 0.1

                # Initialize the SOAP descriptor with dynamic r_cut
                soap = SOAP(species=["H"], periodic=False, r_cut=r_cut_dynamic, n_max=3, l_max=3, sigma=0.1)

                # Create an ASE Atoms object including the current atom and its neighbors
                positions = atom_coords.numpy()[[i] + connected_atoms]
                numbers = np.ones(len(positions))
                system = Atoms(numbers=numbers, positions=positions)

                # Calculate SOAP descriptor for the current atom
                soap_descriptor = soap.create(system)[0]  # [0] to get the descriptor for the central atom

                # Append to the list
                soap_descriptors_list.append(soap_descriptor)

            # Convert list of SOAP descriptors to a torch tensor
            soap_descriptors_tensor = torch.tensor(np.vstack(soap_descriptors_list), dtype=torch.float32)

            # Concatenate the SOAP descriptors with the 'x' tensor
            data['x'] = torch.cat((features_x, soap_descriptors_tensor), dim=1)

            # Delete the 'atom_coords' from the data
            del data['atom_coords']

            # Save the updated data object
            output_file_path = os.path.join(output_directory, filename)
            torch.save(data, output_file_path)

    print(f"All local SOAP descriptors calculated, appended to 'x' and saved to {output_directory}.")

if __name__ == "__main__":

    input_directory = '/content/drive/MyDrive/protein-DATA/sample-normalized'
    output_directory = '/content/drive/MyDrive/protein-DATA/sample-atomic-encoded'
    
    soap_local(input_directory, output_directory)
