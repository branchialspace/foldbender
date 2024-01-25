# Foldseek Multiclass Classification Labels
import pandas as pd
import os
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def filter_encode_clusters(cluster_dict, unlisted_files, input_directory):
    # Filter cluster members based on files in the input directory
    available_files = set([os.path.splitext(f)[0] for f in os.listdir(input_directory) if f.endswith('.pt')])
    for cluster in list(cluster_dict.keys()):
        cluster_dict[cluster] = [member for member in cluster_dict[cluster] if member in available_files]
        if len(cluster_dict[cluster]) == 0:
            del cluster_dict[cluster]

    # Combine small clusters and unlisted files
    combined_cluster = []
    clusters_to_remove = []
    for cluster, members in cluster_dict.items():
        if len(members) < 3:
            combined_cluster.extend(members)
            clusters_to_remove.append(cluster)

    # Add unlisted files to the combined cluster
    combined_cluster.extend(unlisted_files)

    # Remove the small clusters and add the 'combined' cluster
    for cluster in clusters_to_remove:
        cluster_dict.pop(cluster)

    if combined_cluster:
        cluster_dict['combined'] = combined_cluster

    # Manually assign 0 to the 'combined' cluster
    encoded_cluster_dict = {'combined': 0}

    # Prepare remaining clusters for label encoding
    remaining_clusters = {k: v for k, v in cluster_dict.items() if k != 'combined'}

    # Encode remaining clusters starting from 1
    if remaining_clusters:
        label_encoder = LabelEncoder()
        encoded_labels = label_encoder.fit_transform(list(remaining_clusters.keys())) + 1
        encoded_cluster_dict.update({cluster: label for cluster, label in zip(remaining_clusters.keys(), encoded_labels)})

    # Map each member file to its cluster label
    file_to_label_map = {}
    for cluster, members in cluster_dict.items():
        for member in members:
            file_to_label_map[member] = encoded_cluster_dict[cluster]

    return file_to_label_map

def foldseek_multiclass_labels(input_directory, tsv_file_path):
    # Read and process TSV
    df = pd.read_csv(tsv_file_path, sep='\t', header=None)
    cluster_dict = {}
    listed_files = set()
    for cluster, member in zip(df[0], df[1]):
        base_name = os.path.splitext(member)[0]  # Strip the extension
        cluster_dict.setdefault(cluster, []).append(base_name)
        listed_files.add(base_name)

    # Identify unlisted files
    unlisted_files = [os.path.splitext(f)[0] for f in os.listdir(input_directory) if f.endswith('.pt') and os.path.splitext(f)[0] not in listed_files]

    # Encode and filter clusters
    file_to_label_map = filter_encode_clusters(cluster_dict, unlisted_files, input_directory)

    # Modify data objects
    for filename in tqdm(os.listdir(input_directory), desc="Processing Files"):
        if filename.endswith('.pt'):
            base_name = os.path.splitext(filename)[0]
            data_path = os.path.join(input_directory, filename)
            data = torch.load(data_path)

            if base_name in file_to_label_map:
                cluster_label = file_to_label_map[base_name]
                data.y = torch.tensor([cluster_label], dtype=torch.long)
                torch.save(data, data_path)

if __name__ == "__main__":

    tsv_file_path = '/content/drive/MyDrive/protein-DATA/res_cluster.tsv'
    input_directory = '/content/41k_prot_foldseek'

    foldseek_multiclass_labels(input_directory, tsv_file_path)
