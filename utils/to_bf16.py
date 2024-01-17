import os
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def convert_to_bfloat16(directory):
    # List all .pt files in the directory
    files = [f for f in os.listdir(directory) if f.endswith('.pt')]

    for filename in tqdm(files, desc="Converting files"):
        file_path = os.path.join(directory, filename)

        # Load the PyG data object
        data = torch.load(file_path)

        # Check if the object is an instance of PyTorch Geometric Data
        if isinstance(data, Data):
            # Convert all tensor attributes to bfloat16, except edge_index
            # Modify to omit y tensors as well for classification tasks
            for key, value in data:
                if torch.is_tensor(value) and key != 'edge_index':
                    data[key] = value.to(torch.bfloat16)

            # Save the converted data object
            torch.save(data, file_path)
        else:
            tqdm.write(f"{filename} is not a PyTorch Geometric data object.")
          
if __name__ == "__main__":

    convert_to_bfloat16('/content/41k_prot_foldseek')
