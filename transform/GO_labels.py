# Gene Ontology Function Annotation Labels
import torch
import pandas as pd
import os
from torch_geometric.data import Data

def go_labels(input_directory, output_directory, train_terms):
    os.makedirs(output_directory, exist_ok=True)
    # Load the train_terms.tsv file and parse it
    df = pd.read_csv(train_terms, sep='\t')
    df.sort_values(by='term', inplace=True)

    # List all files in the input directory
    all_files = os.listdir(input_directory)

    # Identify all unique terms associated with the files in the directory
    entries_in_directory = [filename.split('.')[0] for filename in all_files if filename.endswith('.pt')]
    unique_terms = df[df['EntryID'].isin(entries_in_directory)]['term'].unique()
    term_to_index = {term: i for i, term in enumerate(unique_terms)}

    # Process each file in the directory
    for filename in all_files:
        if filename.endswith('.pt'):
            entry_id = filename.split('.')[0]  # Assuming the file names are just EntryID.pt

            # If the filename (without .pt) exists in the df's EntryID column
            if entry_id in df['EntryID'].values:
                # Load the .pt file from the local directory as a PyG data object
                data_obj = torch.load(os.path.join(input_directory, filename))

                # Create a binary vector for the terms associated with this entry
                y = torch.zeros(len(unique_terms), dtype=torch.float32)
                entry_terms = df[df['EntryID'] == entry_id]['term'].values
                for term in entry_terms:
                    index = term_to_index[term]
                    y[index] = 1
                data_obj.y = y

                # Save the modified data object to the output directory
                torch.save(data_obj, os.path.join(output_directory, filename))

if __name__ == "__main__":
    
    input_directory = "/content/drive/MyDrive/protein-DATA/sample-atomic-encoded"
    output_directory = "/content/drive/MyDrive/protein-DATA/sample-encoded-labeled"
    train_terms = "/content/drive/MyDrive/cafa-5-protein-function-prediction/Train/train_terms.tsv"
    
    go_labels(input_directory, output_directory, train_terms)
