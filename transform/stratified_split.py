# Multi-label stratified split for train, val, test sets
import os
import shutil
import torch
from collections import Counter
from sklearn.model_selection import train_test_split

def stratified_split(input_directory, indices_file_path):
    # Load all files in the input directory and create a mapping to their indices
    file_list = os.listdir(input_directory)
    file_path_to_index = {os.path.join(input_directory, file): idx for idx, file in enumerate(file_list)}

    # Extract labels for splitting purposes
    label_representations = []
    for file in file_list:
        data = torch.load(os.path.join(input_directory, file))
        label_indices = data.y.nonzero(as_tuple=True)[0]
        label_representations.append(tuple(sorted(label_indices.tolist())))

    # Function to determine if stratification is possible
    def can_stratify(labels):
        label_counts = Counter(labels)
        return all(count > 1 for count in label_counts.values())

    # If we can stratify based on the labels, we do. Otherwise, we split randomly.
    split_args = dict(test_size=0.2, random_state=42)
    if can_stratify(label_representations):
        split_args["stratify"] = label_representations

    train_indices, temp_test_indices, _, _ = train_test_split(range(len(file_list)), label_representations, **split_args)

    temp_label_representations = [label_representations[i] for i in temp_test_indices]
    split_args_val_test = dict(test_size=0.5, random_state=42)
    if can_stratify(temp_label_representations):
        split_args_val_test["stratify"] = temp_label_representations

    val_indices, test_indices = train_test_split(temp_test_indices, **split_args_val_test)
    # Save indices to a .pt file
    indices_dict = {
        'train': train_indices,
        'valid': val_indices,
        'test': test_indices
    }

    torch.save(indices_dict, indices_file_path)

    # Rename files based on the split indices
    for split, indices in indices_dict.items():
        for idx in indices:
            original_file_path = os.path.join(input_directory, file_list[idx])
            new_file_name = f"{idx}.pt"
            new_file_path = os.path.join(input_directory, new_file_name)
            shutil.move(original_file_path, new_file_path)

    return indices_dict

input_directory = '/content/drive/MyDrive/protein-DATA/sample-final'
indices_file_path = '/content/drive/MyDrive/protein-DATA/dataset-indices.pt'
indices_dict = stratified_split(input_directory, indices_file_path)
