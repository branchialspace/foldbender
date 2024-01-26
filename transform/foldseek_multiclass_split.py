# Foldseek Multiclass Stratified Split
import os
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter


def foldseek_multiclass_split(input_directory, valid_size=0.2, test_size=0.5, random_state=42):

    # Define the path for saving indices in the parent directory of input_directory
    indices_file_path = os.path.join(os.path.dirname(input_directory), os.path.basename(input_directory) + "_split_indices.pt")

    # Gather all .pt files
    file_list = [f for f in os.listdir(input_directory) if f.endswith('.pt')]
    file_paths = [os.path.join(input_directory, f) for f in file_list]

    # Extract y attributes for stratification
    labels = []
    for file_path in file_paths:
        data = torch.load(file_path)
        labels.append(data.y.item()) # assuming y is a scalar label

    # Stratified split for train and valid_test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=random_state)
    train_indices, valid_test_indices = next(sss.split(file_paths, labels))

    # Count labels in valid_test set to find classes with only one member
    valid_test_labels = [labels[i] for i in valid_test_indices]
    label_counts = Counter(valid_test_labels)

    # Convert train_indices and valid_test_indices to lists for manipulation
    train_indices = train_indices.tolist()
    valid_test_indices = valid_test_indices.tolist()

    # Adjust valid_test_indices: move classes with only one member left to the validation set
    adjusted_valid_test_indices = []
    for idx in valid_test_indices:
        if label_counts[labels[idx]] == 1:
            train_indices.append(idx)
        else:
            adjusted_valid_test_indices.append(idx)

    # Further stratified split the valid_test set into validation and test sets
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    valid_indices, test_indices = next(sss.split([file_paths[i] for i in adjusted_valid_test_indices], [labels[i] for i in adjusted_valid_test_indices]))

    # Assign file names to train, validation, and test sets based on the split indices
    indices_dict = {
        'train': [file_list[idx] for idx in train_indices],
        'valid': [file_list[idx] for idx in valid_indices],
        'test': [file_list[idx] for idx in test_indices]
    }

    # Save indices to a .pt file
    torch.save(indices_dict, indices_file_path)

    for key in indices_dict.keys():
        print(f"{key}: {len(indices_dict[key])} files")

    return indices_dict


if __name__ == "__main__":
  
    input_directory = '/content/41k_prot_foldseek'
  
    indices_dict = foldseek_multiclass_split(input_directory)
