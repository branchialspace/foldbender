# ESM-2 3B 2560-dim protein language model final layer embeddings as labels 
import numpy as np
import torch
import os
from torch_geometric.data import Data
from tqdm import tqdm

def esm2_labels(embeddings_path, sequence_ids_path, input_dir):
    # Load numpy arrays from the given file paths
    embeddings = np.load(embeddings_path)
    sequence_ids = np.load(sequence_ids_path)

    # Create a dictionary for sequence ID to embedding mapping
    embedding_dict = {seq_id: embedding for seq_id, embedding in zip(sequence_ids, embeddings)}

    # Iterate over the protein files in the input directory with a progress bar
    for filename in tqdm(os.listdir(input_dir), desc="Assigning ESM2 embedding labels as y"):
        file_path = os.path.join(input_dir, filename)
        seq_id = filename.split('.')[0]

        if seq_id in embedding_dict:
            # Load the PyG data object
            pyg_data_object = torch.load(file_path)

            # Add the embedding as an attribute 'y'
            pyg_data_object.y = torch.tensor(embedding_dict[seq_id], dtype=torch.float16)

            # Save the updated PyG object back to the same file
            torch.save(pyg_data_object, file_path)
        else:
            print(f"Embedding not found for sequence ID: {seq_id}")

    print("Embeddings added to all corresponding PyG data objects in the input directory.")

if __name__ == "__main__":

    embeddings_path = 'path_to_embeddings.npy'
    sequence_ids_path = 'path_to_sequence_ids.npy'
    input_dir = 'path_to_input_directory'

    esm2_labels(embeddings_path, sequence_ids_path, input_dir)
