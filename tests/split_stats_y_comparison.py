import os
import torch
from collections import Counter
from torch_geometric.data import Data


def analyze_directory(directory_path, split_file_path):
    # Initialize counters and lists
    y_values = []
    file_names = []
    files_with_missing_y = []

    # Analyze each file in the directory
    for file in os.listdir(directory_path):
        if file.endswith('.pt'):
            file_path = os.path.join(directory_path, file)
            data = torch.load(file_path)
            if isinstance(data, Data):
                file_names.append(file)
                if hasattr(data, 'y') and data.y is not None:
                    y_values.append(int(data.y))
                else:
                    files_with_missing_y.append(file)

    # Count unique y values
    y_counts = Counter(y_values)
    unique_y_values = list(y_counts.keys())

    print(f"Total number of unique y values: {len(unique_y_values)}")
    print(f"All possible integer values of y: {unique_y_values}")
    print(f"Total count of each unique y value: {y_counts}")
    print(f"Files with missing y attribute: {files_with_missing_y}")

    # Analyze the split file
    split_dict = torch.load(split_file_path)
    for split, files in split_dict.items():
        split_y_values = [int(torch.load(os.path.join(directory_path, f)).y) for f in files]
        unique_split_y_values = set(split_y_values)
        print(f"Total number of different categories in {split} split: {len(unique_split_y_values)}")

    # Check for missing files in splits
    all_split_files = set(sum(split_dict.values(), []))
    missing_in_splits = set(file_names) - all_split_files
    missing_in_directory = all_split_files - set(file_names)
    print(f"Files in directory but missing in splits: {missing_in_splits}")
    print(f"Files in splits but missing in directory: {missing_in_directory}")


directory_path = '/content/41k_prot_go'
split_file_path = '/content/drive/MyDrive/protein-DATA/41k_prot_foldseek_split_indices.pt'
analyze_directory(directory_path, split_file_path)
