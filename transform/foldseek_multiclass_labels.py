# Foldseek Multiclass Classification Labels
import pandas as pd
import os
import torch
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def filter_encode_clusters(cluster_dict, unlisted_files):
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

    # Encode all clusters including the 'combined' cluster
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(list(cluster_dict.keys()))

    encoded_cluster_dict = {label: cluster_dict[cluster] for label, cluster in zip(encoded_labels, cluster_dict.keys())}

    return encoded_cluster_dict

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
    encoded_cluster_dict = filter_encode_clusters(cluster_dict, unlisted_files)

    # Modify data objects
    for filename in tqdm(os.listdir(input_directory), desc="Processing Files"):
        if filename.endswith('.pt'):
            base_name = os.path.splitext(filename)[0]
            data_path = os.path.join(input_directory, filename)
            data = torch.load(data_path)

            for cluster_label, members in encoded_cluster_dict.items():
                if base_name in members:
                    data.y = torch.tensor([cluster_label], dtype=torch.long)
                    torch.save(data, data_path)
                    break

if __name__ == "__main__":

    tsv_file_path = '/content/drive/MyDrive/protein-DATA/res_cluster.tsv'
    directory_path = '/content/41k_prot_foldseek'

    foldseek_multiclass_labels(input_directory, tsv_file_path)
