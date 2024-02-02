# Foldseek Multitarget Regression Labels
import torch
import pandas as pd
import os
from torch_geometric.data import Data
from tqdm import tqdm


def foldseek_regression_labels(input_directory, foldseek_labels):
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

    # Process each file in the directory using tqdm
    for filename in tqdm(all_files, desc="Assigning Foldseek cluster similarity score regression labels as y"):
        if filename.endswith('.pt'):
            entry_id = filename.split('.')[0]  # Assuming the file names are just EntryID.pt
            file_path = os.path.join(input_directory, filename)
            data_obj = torch.load(file_path)

            # Initialize y vector for all unique terms
            y = torch.zeros(len(unique_terms), dtype=torch.float16)

            if entry_id in df['EntryID'].values:
                # Create a vector for the terms associated with this entry
                entry_df = df[df['EntryID'] == entry_id]
                for _, row in entry_df.iterrows():
                    index = term_to_index[row['Term']]
                    y[index] = row['Value']
            # If the entry_id is not in foldseek_labels, y remains a zero vector
            data_obj.y = y
            # Save the modified data object back to the same location
            torch.save(data_obj, file_path)
              
if __name__ == "__main__":
    
    input_directory = "/content/drive/MyDrive/protein-DATA/sample"
    foldseek_labels = ".tsv"
    
    foldseek_regression_labels(input_directory, foldseek_labels)
