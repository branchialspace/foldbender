# Multi-label stratified split for train, val, test sets
import os
import shutil
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def stratified_split(input_directory, n_splits=3):
    # Define the path for saving indices
    indices_file_path = os.path.join(os.path.dirname(input_directory), f"{os.path.basename(input_directory)}_split_indices.pt")

    # Load all files in the input directory and create a mapping to their indices
    file_list = os.listdir(input_directory)

    # Extract labels for splitting purposes
    label_representations = []
    for file in file_list:
        data = torch.load(os.path.join(input_directory, file))
        label_indices = data.y.nonzero(as_tuple=True)[0]
        label_representations.append(tuple(sorted(label_indices.tolist())))

    # Convert label representations to a format suitable for MultilabelStratifiedKFold
    unique_labels = sorted(set(sum(label_representations, ())))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    multilabel_format = torch.zeros(len(file_list), len(unique_labels))
    for i, labels in enumerate(label_representations):
        for label in labels:
            multilabel_format[i, label_to_index[label]] = 1

    # Use MultilabelStratifiedKFold for splitting
    mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    indices = list(mskf.split(multilabel_format, multilabel_format))
    
    # Assign indices to train, validation, and test sets
    indices_dict = {
        'train': indices[0][0],  # First fold for training
        'valid': indices[1][0],  # Second fold for validation
        'test': indices[2][0]    # Third fold for testing
    }

    # Save indices to a .pt file
    torch.save(indices_dict, indices_file_path)

    # Rename files based on the split indices
    for split, indices in indices_dict.items():
        for idx in indices:
            original_file_path = os.path.join(input_directory, file_list[idx])
            new_file_name = f"{idx}.pt"
            new_file_path = os.path.join(input_directory, new_file_name)
            shutil.move(original_file_path, new_file_path)

    return indices_dict

if __name__ == "__main__":
    
    input_directory = '/content/drive/MyDrive/protein-DATA/sample-final'
    
    indices_dict = stratified_split(input_directory)
