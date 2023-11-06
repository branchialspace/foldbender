import torch
from dscribe.descriptors import SOAP
from torch_geometric.data import Data
import os

def soap_local(directory):
    # Initialize the SOAP descriptor
    soap = SOAP(species=["x", "y"], periodic=False, rcut=3.0, nmax=3, lmax=3, sigma=0.1)

    # Iterate over the .pt files in the directory
    for filename in os.listdir(directory):
        if filename.endswith('.pt'):
            # Load the PyTorch Geometric Data object
            data = torch.load(os.path.join(directory, filename))

            # Retrieve the atom coordinates
            atom_coords = data['atom_coords']

            # This will hold the local SOAP descriptors
            local_descriptors = []

            # Calculate SOAP descriptor for each atom
            for i in range(len(atom_coords)):
                # Define the current atom as "x" and all others as "y"
                species = ["y"] * len(atom_coords)
                species[i] = "x"  # The current atom is of type "x"

                # Calculate the descriptor for the current atom
                descriptor = soap.create(atom_coords, species=species)
                local_descriptors.append(descriptor[i])  # Only append the descriptor for the central atom

            # Add the local descriptors to the data object
            data['local_xyz'] = torch.tensor(local_descriptors)

            # Save the modified data object back to disk
            torch.save(data, os.path.join(directory, filename))
            
    print("All local SOAP descriptors calculated and saved.")

# Usage
# soap_local('/path/to/your/directory')
