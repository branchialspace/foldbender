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
    edge_set = set()
    total_filtered_edges = 0

    # Add all edges that pass the filter to the set
    for i, (u, v) in enumerate(edge_index.t()):
        if edge_filter(edge_attr[i]):
            edge_set.add((u.item(), v.item()))
            total_filtered_edges += 1

    symmetric_count = 0
    asymmetric_values = []

    # Check for the presence of both (u, v) and (v, u)
    for (u, v) in edge_set:
        if (v, u) in edge_set:
            symmetric_count += 1
        else:
            asymmetric_values.append(edge_attr[i])

    return symmetric_count, total_filtered_edges, asymmetric_values

def analyze_directory(directory, sample_size):
    files = os.listdir(directory)
    selected_files = random.sample(files, min(sample_size, len(files)))

    bond_sym_results = []
    pae_sym_results = []
    asymmetry_bond = []
    asymmetry_pae = []

    for file in selected_files:
        data = torch.load(os.path.join(directory, file))
        bond_sym, bond_total, bond_asymmetry = check_symmetry(data.edge_index, data.edge_attr, is_bond)
        pae_sym, pae_total, pae_asymmetry = check_symmetry(data.edge_index, data.edge_attr, is_pae)
        bond_sym_results.append((bond_sym, bond_total))
        pae_sym_results.append((pae_sym, pae_total))
        asymmetry_bond.extend(bond_asymmetry)
        asymmetry_pae.extend(pae_asymmetry)

    return bond_sym_results, pae_sym_results, asymmetry_bond, asymmetry_pae

def bin_asymmetry_data(asymmetry_data):
    if not asymmetry_data:
        return [0, 0, 0, 0]  # Return zeros if there are no asymmetric data

    relative_differences = []
    for attr1, attr2 in asymmetry_data:
        # Compare corresponding positions in attr1 and attr2
        relative_diff = torch.abs(attr1 - attr2) / torch.clamp(torch.max(torch.abs(attr1), torch.abs(attr2)), min=1e-6)
        relative_differences.extend(relative_diff.numpy())  # Flattening and converting to numpy

    if not relative_differences:
        return [0, 0, 0, 0]

    relative_differences = np.array(relative_differences)

    # Define bins for the quartiles
    bins = [0.25, 0.5, 0.75, 1.0]

    # Digitize the relative differences into bins
    binned_data = np.digitize(relative_differences, bins, right=False)

    # Calculate the percentage of total for each bin
    bin_counts = np.bincount(binned_data, minlength=4)
    percentages = (bin_counts / bin_counts.sum()) * 100  # Convert to percentages

    return percentages.tolist()

def execute_analysis(directory, sample_size):
    bond_sym_results, pae_sym_results, asymmetry_bond, asymmetry_pae = analyze_directory(directory, sample_size)

    # Process the asymmetry data
    binned_bond_asymmetry = bin_asymmetry_data(asymmetry_bond)
    binned_pae_asymmetry = bin_asymmetry_data(asymmetry_pae)

    return {
        "Bond Symmetry Counts": bond_sym_results,
        "PAE Symmetry Counts": pae_sym_results,
        "Binned Bond Asymmetry Distribution": binned_bond_asymmetry,
        "Binned PAE Asymmetry Distribution": binned_pae_asymmetry
    }

directory = '/content/drive/MyDrive/protein-DATA/sample-final'
sample_size = 7

results = execute_analysis(directory, sample_size)

print("Results:")
for key, value in results.items():
    print(f"{key}: {value}")
