# ESM-2 3B 2560-dim protein language model final layer embeddings as labels 
import numpy as np
import torch
import os
from torch_geometric.data import Data

def esm2_labels(embeddings_path, sequence_ids_path, input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Load numpy arrays from the given file paths
    embeddings = np.load(embeddings_path)
    sequence_ids = np.load(sequence_ids_path)

    # Create a dictionary for sequence ID to embedding mapping
    embedding_dict = {seq_id: embedding for seq_id, embedding in zip(sequence_ids, embeddings)}

    # Iterate over the protein files in the input directory
    for filename in os.listdir(input_dir):
        file_path = os.path.join(input_dir, filename)
        seq_id = filename.split('.')[0]

        if seq_id in embedding_dict:
            # Load the PyG data object
            pyg_data_object = torch.load(file_path)

            # Add the embedding as an attribute 'y'
            pyg_data_object.y = torch.tensor(embedding_dict[seq_id], dtype=torch.float32)

            # Save the updated PyG object to the output directory
            output_file_path = os.path.join(output_dir, filename)
            torch.save(pyg_data_object, output_file_path)
        else:
            print(f"Embedding not found for sequence ID: {seq_id}")

    print("Embeddings added to all corresponding PyG data objects in the output directory.")

if __name__ == "__main__":

    embeddings_path = 'path_to_embeddings.npy'
    sequence_ids_path = 'path_to_sequence_ids.npy'
    input_dir = 'path_to_input_directory'
    output_dir = 'path_to_output_directory'

    esm2_labels(embeddings_path, sequence_ids_path, input_dir, output_dir)
