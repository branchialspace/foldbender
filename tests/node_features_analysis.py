# Check for missing DSSP node features
import os
import pickle
import networkx as nx
import random
from collections import defaultdict

def check_missing_keys(directory, sample_size=30):
    missing_key_files = 0
    file_missing_nodes = {}
    all_keys = set()
    files = [f for f in os.listdir(directory) if f.endswith('.pkl')]
    selected_files = random.sample(files, min(sample_size, len(files)))
    for filename in selected_files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as file:
            graph = pickle.load(file)
            if graph.nodes:
                random_node = random.choice(list(graph.nodes))
                all_keys.update(graph.nodes[random_node].keys())

    for filename in files:
        file_path = os.path.join(directory, filename)
        with open(file_path, 'rb') as file:
            graph = pickle.load(file)
            missing_keys_in_file = defaultdict(int)
            for node in graph.nodes:
                for key in all_keys:
                    if key not in graph.nodes[node]:
                        missing_keys_in_file[key] += 1
            if missing_keys_in_file:
                missing_key_files += 1
                file_missing_nodes[filename] = dict(missing_keys_in_file)

    return missing_key_files, file_missing_nodes, all_keys

directory_path = "/content/41k_NX"

total_missing, files_with_missing, keys = check_missing_keys(directory_path)

print(f"Expected keys: {keys}")
print(f"Total files with missing keys: {total_missing}")
for file, missing in files_with_missing.items():
    missing_info = ', '.join([f"'{k}': {v} nodes" for k, v in missing.items()])
    print(f"{file}: {missing_info}")
