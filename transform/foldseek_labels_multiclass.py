# Foldseek Multiclass Classification Labels
import torch
import pandas as pd
import os
from torch_geometric.data import Data
import tqdm


def foldseek_labels_multiclass(input_directory, foldseek_labels):
    # Load the foldseek_labels.tsv file and parse it
    df = pd.read_csv(foldseek_labels, sep='\t', header=None, names=['Term', 'EntryID_with_extra', 'Value'])
    df['EntryID'] = df['EntryID_with_extra'].apply(lambda x: x.split('.')[0])

    # Identify all unique terms
    unique_terms = df['Term'].unique()
    term_to_index = {term: i for i, term in enumerate(unique_terms)}

    # List all files in the input directory
    all_files = os.listdir(input_directory)

    # Process each file in the directory using tqdm
    for filename in tqdm(all_files, desc="Creating y:"):
        if filename.endswith('.pt'):
            entry_id = filename.split('.')[0]  # Assuming the file names are just EntryID.pt
            file_path = os.path.join(input_directory, filename)
            data_obj = torch.load(file_path)

            # Initialize y vector for all unique terms
            y = torch.zeros(len(unique_terms), dtype=torch.bool)

            if entry_id in df['EntryID'].values:
                # Filter the dataframe for the current entry
                entry_df = df[df['EntryID'] == entry_id]

                # Find the term with the highest value for this entry
                max_row = entry_df.loc[entry_df['Value'].idxmax()]
                max_term_index = term_to_index[max_row['Term']]

                # Set the corresponding index in y to 1
                y[max_term_index] = 1
            # If the entry_id is not in foldseek_labels, y remains a zero vector
            data_obj.y = y
            # Save the modified data object back to the same location
            torch.save(data_obj, file_path)


if __name__ == "__main__":
    
    input_directory = "/content/drive/MyDrive/protein-DATA/sample"
    foldseek_labels = ".tsv"
    
    foldseek_labels_multiclass(input_directory, foldseek_labels)


