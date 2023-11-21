# Analyze the distribution of PAE edges values and directedness to inform dimensionality reduction strategy
import os
import random
import torch
from torch_geometric.data import Data
import numpy as np

def is_bond(edge_attr):
    return edge_attr[-1] == 0

def is_pae(edge_attr):
    return edge_attr[-1] != 0

def check_symmetry(edge_index, edge_attr, edge_filter):
    edge_dict = {}
    asymmetric_values = []

    for i, (u, v) in enumerate(edge_index.t()):
        if edge_filter(edge_attr[i]):
            edge_dict[(u.item(), v.item())] = edge_attr[i]

    for (u, v), attr in edge_dict.items():
        if (v, u) in edge_dict and not torch.equal(attr, edge_dict[(v, u)]):
            asymmetric_values.append(torch.abs(attr - edge_dict[(v, u)]))

    return len(asymmetric_values), asymmetric_values

def bin_asymmetry_data(asymmetry_data):
    if not asymmetry_data:
        return []

    values = torch.cat(asymmetry_data, dim=0).numpy()
    quantiles = np.percentile(values, [25, 50, 75, 100])
    binned_data = np.digitize(values, quantiles)

    return binned_data

def analyze_directory(directory, sample_size):
    files = os.listdir(directory)
    selected_files = random.sample(files, min(sample_size, len(files)))

    bond_dir_results = []
    pae_dir_results = []
    asymmetry_bond = []
    asymmetry_pae = []

    for file in selected_files:
        data = torch.load(os.path.join(directory, file))
        bond_count, bond_asymmetry = check_symmetry(data.edge_index, data.edge_attr, is_bond)
        pae_count, pae_asymmetry = check_symmetry(data.edge_index, data.edge_attr, is_pae)
        bond_dir_results.append(bond_count / len(data.edge_attr))
        pae_dir_results.append(pae_count / len(data.edge_attr))
        asymmetry_bond.extend(bond_asymmetry)
        asymmetry_pae.extend(pae_asymmetry)

    return bond_dir_results, pae_dir_results, asymmetry_bond, asymmetry_pae


def execute_analysis(directory, sample_size):
    bond_dir_results, pae_dir_results, asymmetry_bond, asymmetry_pae = analyze_directory(directory, sample_size)

    # Process the asymmetry data
    binned_bond_asymmetry = bin_asymmetry_data(asymmetry_bond)
    binned_pae_asymmetry = bin_asymmetry_data(asymmetry_pae)

    return {
        "Bond Directedness": bond_dir_results,
        "PAE Directedness": pae_dir_results,
        "Binned Bond Asymmetry Distribution": binned_bond_asymmetry,
        "Binned PAE Asymmetry Distribution": binned_pae_asymmetry
    }

directory = '/content/drive/MyDrive/protein-DATA/sample-final'
sample_size = 10

results = execute_analysis(directory, sample_size)

print("Results:")
for key, value in results.items():
    print(f"{key}: {value}")
