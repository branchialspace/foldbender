# Foldseek Multiclass Stratified Split
import os
import torch
from torch_geometric.data import Data
from sklearn.model_selection import StratifiedShuffleSplit


def foldseek_multiclass_split(input_directory, test_size=0.2, valid_size=0.5, random_state=42):
    # Define the path for saving indices
    indices_file_path = os.path.join(os.path.dirname(input_directory), f"{os.path.basename(input_directory)}_split_indices.pt")
  
    # Load data and labels
    file_list = []
    class_labels = []
    for file in os.listdir(data_dir):
        if file.endswith('.pt'):
            file_path = os.path.join(data_dir, file)
            data_object = torch.load(file_path)
            file_list.append(file)
            class_labels.append(data_object.y.item())  # Assuming 'y' is a scalar label

    # First stratified shuffle split (train + validation/test)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_indices, test_val_indices = next(sss.split(file_list, class_labels))

    # Second stratified shuffle split (validation/test)
    sss_val_test = StratifiedShuffleSplit(n_splits=1, test_size=valid_size, random_state=random_state)
    valid_indices, test_indices = next(sss_val_test.split([file_list[i] for i in test_val_indices], [class_labels[i] for i in test_val_indices]))

    # Assign file names to train, validation, and test sets based on the split indices
    indices_dict = {
        'train': [file_list[idx] for idx in train_indices],
        'valid': [file_list[test_val_indices[idx]] for idx in valid_indices],
        'test': [file_list[test_val_indices[idx]] for idx in test_indices]
    }

    # Save indices to a .pt file
    torch.save(indices_dict, indices_file_path)

    for key in indices_dict.keys():
        print(f"{key}: {len(indices_dict[key])} files")

    return indices_dict

if __name__ == "__main__":
  
    input_directory = '/content/drive/MyDrive/protein-DATA/sample-final'
  
    indices_dict = foldseek_multiclass_split(input_directory)
