# Foldseek Multiclass Classification Labels
import torch
import pandas as pd
import numpy as np
import os
from torch_geometric.data import Data
from tqdm import tqdm


def foldseek_labels_multiclass(input_directory, foldseek_labels):
    # Load the foldseek_labels.tsv file and parse it
    df = pd.read_csv(foldseek_labels, sep='\t', header=None, names=['Term', 'EntryID_with_extra', 'Value'])
    df['EntryID'] = df['EntryID_with_extra'].apply(lambda x: x.split('.')[0])

    # Identify all unique terms
    unique_terms = df['Term'].unique()
    
    # Define a unique label for entries with 0 terms
    no_term_label = 'NoTerm'

    # Add the unique label to the list of terms
    unique_terms_with_no_term = np.append(unique_terms, no_term_label)

    term_to_index = {term: i for i, term in enumerate(unique_terms_with_no_term)}

    # List all files in the input directory
    all_files = os.listdir(input_directory)

    # Process each file in the directory using tqdm
    for filename in tqdm(all_files, desc="Creating y:"):
        if filename.endswith('.pt'):
            entry_id = filename.split('.')[0]  # Assuming the file names are just EntryID.pt
            file_path = os.path.join(input_directory, filename)
            data_obj = torch.load(file_path)

            # Initialize y with the unique label for 0 terms
            y = torch.tensor([term_to_index[no_term_label]], dtype=torch.long)

            if entry_id in df['EntryID'].values:
                # Filter the dataframe for the current entry
                entry_df = df[df['EntryID'] == entry_id]

                # Find the term with the highest value for this entry
                max_row = entry_df.loc[entry_df['Value'].idxmax()]
                max_term_index = term_to_index[max_row['Term']]

                # Set y to the index of the term with the highest value
                y = torch.tensor([max_term_index], dtype=torch.long)

            data_obj.y = y
            # Save the modified data object back to the same location
            torch.save(data_obj, file_path)


if __name__ == "__main__":
    
    input_directory = "/content/drive/MyDrive/protein-DATA/sample"
    foldseek_labels = ".tsv"
    
    foldseek_labels_multiclass(input_directory, foldseek_labels)


