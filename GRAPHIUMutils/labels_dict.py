# Implement Labels Dict SingleTaskDataset
import torch
from google.cloud import storage
import pandas as pd
from torch_geometric.data import Data
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/content/drive/MyDrive/instant-tape-******.json"
storage_client = storage.Client()

train_terms = "/train_terms.tsv"

# Load the train_terms.tsv file and parse it
df = pd.read_csv(train_terms, sep='\t')

# Sort the DataFrame by EntryID alphanumerically
df.sort_values(by='term', inplace=True)

# Connect to the GCS bucket and retrieve the filenames
client = storage.Client()
bucket = client.get_bucket('pyg-molecular-encoded')
blobs = list(bucket.list_blobs())

# Identify all unique terms associated with the files in the bucket
entries_in_bucket = [blob.name.split('.')[0] for blob in blobs if blob.name.endswith('.pkl')]
unique_terms = df[df['EntryID'].isin(entries_in_bucket)]['term'].unique()

# Helper function to create the new Data object with labels
def create_label_data(graph_data, terms):
    num_nodes = graph_data.num_nodes
    edge_index_length = graph_data.edge_index.size(1)

    # Initialize labels with 0 for all terms associated with the files in the bucket
    labels = Data(x=torch.zeros((num_nodes, 1)),
                  edge_index=torch.zeros((2, edge_index_length), dtype=torch.long))

    # Add attributes for each term with default value 0
    for term in unique_terms:
        setattr(labels, term, torch.tensor([0]))

    # Update the labels that are associated with this graph_data to 1
    for term in terms:
        setattr(labels, term, torch.tensor([1]))

    return labels

# Process each file in the bucket
for blob in blobs:
    blob_name = blob.name
    if blob_name.endswith('.pkl'):
        entry_id = blob_name.split('.')[0]  # Assuming the file names are just EntryID.pkl

        # If the blob_name (without .pkl) exists in the df's EntryID column
        if entry_id in df['EntryID'].values:
            # Load the .pkl file from GCS into a PyTorch tensor
            blob.download_to_filename('temp.pkl')
            data_dict = torch.load('temp.pkl')

            terms = df[df['EntryID'] == entry_id]['term'].values

            # Create and update the labels Data object
            data_dict['labels'] = create_label_data(data_dict['graph_with_features'], terms)

            # Save the modified dictionary to a temporary file before uploading it
            torch.save(data_dict, 'temp.pkl')

            # Upload the modified file to the new GCS bucket with the original blob_name
            new_bucket = client.get_bucket('pyg-dict-singletaskdataset')
            new_blob = new_bucket.blob(blob_name)
            new_blob.upload_from_filename('temp.pkl')
