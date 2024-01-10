# Foldseek Labels
import torch
import pandas as pd
import os
from torch_geometric.data import Data

def foldseek_labels(input_directory, foldseek_labels):
    # Load the foldseek_labels.tsv file and parse it
    df = pd.read_csv(foldseek_labels, sep='\t', header=None, names=['Term', 'EntryID_with_extra', 'Value'])
    df['EntryID'] = df['EntryID_with_extra'].apply(lambda x: x.split('.')[0])
    df.sort_values(by='Term', inplace=True)

    # List all files in the input directory
    all_files = os.listdir(input_directory)

    # Identify all unique terms associated with the files in the directory
    entries_in_directory = [filename.split('.')[0] for filename in all_files if filename.endswith('.pt')]
    unique_terms = df[df['EntryID'].isin(entries_in_directory)]['Term'].unique()
    term_to_index = {term: i for i, term in enumerate(unique_terms)}

    # Process each file in the directory
    for filename in all_files:
        if filename.endswith('.pt'):
            entry_id = filename.split('.')[0]  # Assuming the file names are just EntryID.pt

            # If the filename (without .pt) exists in the df's EntryID column
            if entry_id in df['EntryID'].values:
                # Load the .pt file from the local directory as a PyG data object
                file_path = os.path.join(input_directory, filename)
                data_obj = torch.load(file_path)

                # Create a vector for the terms associated with this entry
                y = torch.zeros(len(unique_terms), dtype=torch.float32)
                entry_df = df[df['EntryID'] == entry_id]
                for _, row in entry_df.iterrows():
                    index = term_to_index[row['Term']]
                    y[index] = row['Value']
                data_obj.y = y

                # Save the modified data object back to the same location
                torch.save(data_obj, file_path)
              
if __name__ == "__main__":
    
    input_directory = "/content/drive/MyDrive/protein-DATA/sample"
    train_terms = "/content/drive/MyDrive/cafa-5-protein-function-prediction/Train/train_terms.tsv"
    
    foldseek_labels(input_directory, foldseek_labels)
