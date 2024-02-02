# SOAP 3d-coordinate Positional Encodings
import torch
import numpy as np
import ase
from ase import Atoms
from dscribe.descriptors import SOAP
from torch_geometric.data import Data
import os

def soap_local(input_dir, r_cut=3, n_max=3, l_max=3, sigma=0.1):
    # Iterate over the .pt files in the directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.pt'):
           # Load the PyTorch Geometric Data object
            file_path = os.path.join(input_dir, filename)
            data = torch.load(file_path)

            # Retrieve the atom coordinates and existing 'x' tensor
            atom_coords = data['atom_coords']
            features_x = data['x']
            num_atoms = len(atom_coords)

            # Initialize the SOAP descriptor
            soap = SOAP(species=["H"], periodic=False, r_cut=r_cut, n_max=n_max, l_max=l_max, sigma=sigma)

            # Create an ASE Atoms object
            system = Atoms(numbers=np.ones(num_atoms), positions=atom_coords.numpy())

            # Calculate SOAP descriptor for each atom
            soap_descriptors = soap.create(system)

            # Ensure the SOAP descriptors are in the correct shape for concatenation
            soap_descriptors = torch.tensor(soap_descriptors, dtype=torch.float16)

            # Concatenate the SOAP descriptors with the 'x' tensor
            data['x'] = torch.cat((features_x, soap_descriptors), dim=1)

            # Delete the 'atom_coords' from the data
            del data['atom_coords']

            # Save the updated data object
            torch.save(data, file_path)

    print(f"All local SOAP descriptors calculated, appended to 'x' and saved to {input_dir}.")

if __name__ == "__main__":

    input_dir = '/content/drive/MyDrive/protein-DATA/sample-normalized'
    
    soap_local(input_dir, r_cut=3, n_max=3, l_max=3, sigma=0.1)
