# Foldseek Multiclass Classification Labels
import pandas as pd
import os
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def filter_encode_clusters(cluster_dict, unlisted_files, input_dir):
    # Filter cluster members based on files in the input directory
    available_files = set([os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.pt')])
    for cluster in list(cluster_dict.keys()):
        cluster_dict[cluster] = [member for member in cluster_dict[cluster] if member in available_files]
        if len(cluster_dict[cluster]) == 0:
            del cluster_dict[cluster]

    # Combine small clusters and unlisted files into a single list for deletion
    files_to_delete = []
    clusters_to_remove = []
    for cluster, members in cluster_dict.items():
        if len(members) < 3:
            files_to_delete.extend(members)
            clusters_to_remove.append(cluster)

    # Add unlisted files to the deletion list
    files_to_delete.extend(unlisted_files)

    # Remove the small clusters
    for cluster in clusters_to_remove:
        cluster_dict.pop(cluster)

    # Return the list of files to delete
    return files_to_delete, cluster_dict

def foldseek_multiclass_labels(input_dir, foldseek_targets):
    # Read and process TSV
    df = pd.read_csv(foldseek_targets, sep='\t', header=None)
    cluster_dict = {}
    listed_files = set()
    for cluster, member in zip(df[0], df[1]):
        base_name = os.path.splitext(member)[0]  # Strip the extension
        cluster_dict.setdefault(cluster, []).append(base_name)
        listed_files.add(base_name)

    # Identify unlisted files
    unlisted_files = [os.path.splitext(f)[0] for f in os.listdir(input_dir) if f.endswith('.pt') and os.path.splitext(f)[0] not in listed_files]

    # Get files to delete and filter & encode remaining clusters
    files_to_delete, filtered_cluster_dict = filter_encode_clusters(cluster_dict, unlisted_files, input_dir)

    # Delete files in the combined cluster
    for filename in files_to_delete:
        file_path = os.path.join(input_dir, filename + '.pt')
        if os.path.exists(file_path):
            os.remove(file_path)

    # Prepare clusters for label encoding
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(list(filtered_cluster_dict.keys()))
    encoded_cluster_dict = {cluster: label for cluster, label in zip(filtered_cluster_dict.keys(), encoded_labels)}

    # Map each member file to its cluster label
    file_to_label_map = {}
    for cluster, members in filtered_cluster_dict.items():
        for member in members:
            file_to_label_map[member] = encoded_cluster_dict[cluster]

    # Modify data objects
    for filename in tqdm(os.listdir(input_dir), desc="Assigning Foldseek cluster multiclass labels as y"):
        if filename.endswith('.pt'):
            base_name = os.path.splitext(filename)[0]
            data_path = os.path.join(input_dir, filename)
            data = torch.load(data_path)

            if base_name in file_to_label_map:
                cluster_label = file_to_label_map[base_name]
                data.y = torch.tensor([cluster_label], dtype=torch.long)
                torch.save(data, data_path)

if __name__ == "__main__":
    
    foldseek_targets = '/content/drive/MyDrive/protein-DATA/res_cluster.tsv'
    input_dir = '/content/41k_prot_foldseek'

    foldseek_multiclass_labels(input_dir, foldseek_targets)
