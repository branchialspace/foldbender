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

            # Retrieve the atom coordinates
            atom_coords = data['atom_coords']
            num_atoms = len(atom_coords)

            # This will hold the local SOAP descriptors
            local_descriptors = []

            # Initialize the SOAP descriptor
            soap = SOAP(species=["H"], periodic=False, r_cut=2.5, n_max=2, l_max=3, sigma=0.1)

            # Create an ASE Atoms object
            system = Atoms(numbers=np.ones(num_atoms), positions=atom_coords.numpy())

            # Calculate SOAP descriptor for each atom
            for i in range(num_atoms):
                # Calculate the descriptor for the atom
                descriptor = soap.create(system, centers=[i])
                local_descriptors.append(descriptor[0])

            # Add the local descriptors to the data object and save
            local_descriptors_array = np.array(local_descriptors)
            data['local_soap'] = torch.tensor(local_descriptors_array, dtype=torch.float32)
            data_dict[data_key] = data
            output_file_path = os.path.join(output_directory, filename)
            torch.save(data_dict, output_file_path)

    print(f"All local SOAP descriptors calculated and saved to {output_directory}.")

# Call the function with the input and output directories
soap_local(input_directory, output_directory)
