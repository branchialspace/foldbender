import torch
from dscribe.descriptors import SOAP
from torch_geometric.data import Data
import os

input_directory = '/content/drive/MyDrive/protein-DATA/sample-normalized'
output_directory = '/content/drive/MyDrive/protein-DATA/sample-atomic-encoded'

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
            data['local_soap'] = torch.tensor(local_descriptors, dtype=torch.float32)  # Ensure correct tensor type

            # Update the dictionary with the modified data object
            data_dict[data_key] = data

            # Define the path for the output file
            output_file_path = os.path.join(output_directory, filename)

            # Save the updated dictionary back to disk in the separate directory
            torch.save(data_dict, output_file_path)
            
    print(f"All local SOAP descriptors calculated and saved to {output_directory}.")

soap_local(input_directory, output_directory)
