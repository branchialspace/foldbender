import torch
import numpy as np
import ase
from ase import Atoms
from dscribe.descriptors import SOAP
from torch_geometric.data import Data
import os

input_directory = '/content/drive/MyDrive/protein-DATA/sample-normalized'
output_directory = '/content/drive/MyDrive/protein-DATA/sample-atomic-encoded'

def soap_local(input_directory, output_directory):
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

            # Retrieve the atom coordinates and existing 'x' tensor
            atom_coords = data['atom_coords']
            features_x = data['x']
            num_atoms = len(atom_coords)

            # Initialize the SOAP descriptor
            soap = SOAP(species=["H"], periodic=False, r_cut=2.5, n_max=4, l_max=4, sigma=0.1)

            # Create an ASE Atoms object
            system = Atoms(numbers=np.ones(num_atoms), positions=atom_coords.numpy())

            # Calculate SOAP descriptor for each atom and append it to its corresponding feature vector
            for i in range(num_atoms):
                # Calculate the descriptor for the atom
                descriptor = soap.create(system, centers=[i])
                # Append the descriptor to the corresponding feature vector in 'x'
                features_x[i] = torch.cat((features_x[i], torch.tensor(descriptor, dtype=torch.float32).squeeze(0)))

            # Save the updated data object
            data['x'] = features_x
            data_dict[data_key] = data
            output_file_path = os.path.join(output_directory, filename)
            torch.save(data_dict, output_file_path)

    print(f"All local SOAP descriptors calculated, appended to 'x' and saved to {output_directory}.")

# Call the function with the input and output directories
soap_local(input_directory, output_directory)
