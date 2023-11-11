import torch
import pandas as pd
import os

input_directory = "/content/drive/MyDrive/protein-DATA/sample-atomic-encoded"
output_directory = "/content/drive/MyDrive/protein-DATA/sample-encoded-labeled"
train_terms = "/content/drive/MyDrive/cafa-5-protein-function-prediction/Train/train_terms.tsv"

# Load the train_terms.tsv file and parse it
df = pd.read_csv(train_terms, sep='\t')

# Sort the DataFrame by term alphanumerically
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
            # Load the .pt file from the local directory into a PyTorch tensor
            data_dict = torch.load(os.path.join(input_directory, filename))

            # Create a binary vector for the terms associated with this entry
            y = torch.zeros(len(unique_terms), dtype=torch.float32)
            entry_terms = df[df['EntryID'] == entry_id]['term'].values
            for term in entry_terms:
                index = term_to_index[term]
                y[index] = 1
                
            data_dict['y'] = y

            # Save the modified dictionary to the output directory
            torch.save(data_dict, os.path.join(output_directory, filename))
