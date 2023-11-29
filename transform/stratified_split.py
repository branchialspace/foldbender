# Multi-label stratified split for train, val, test sets
import os
import torch
import zipfile
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold

def stratified_split(input_directory, n_splits=8, zip_io=True):
    # Define the path for saving indices
    indices_file_path = os.path.join(os.path.dirname(input_directory), f"{os.path.basename(input_directory)}_split_indices.pt")

    # Load all files in the input directory
    file_list = os.listdir(input_directory)

    # Extract labels for splitting purposes
    label_representations = []

    for file in file_list:
        # Handle zip files if zip_io is True
        if zip_io and file.endswith('.zip'):
            with tempfile.TemporaryDirectory() as temp_dir:
                # Extract the zip file to the temporary directory
                zip_path = os.path.join(input_directory, file)
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(temp_dir)

                # Assuming there's only one file in the zip, load it
                temp_file = os.listdir(temp_dir)[0]
                data = torch.load(os.path.join(temp_dir, temp_file))
        else:
            # Load data normally for non-zip files
            data = torch.load(os.path.join(input_directory, file))

        # Extract labels for each file
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

    # Combine first 6 folds for training, use 7th for validation and 8th for testing
    train_indices = [idx for fold in indices[:6] for idx in fold[0]]
    valid_indices = indices[6][0]
    test_indices = indices[7][0]

    # Assign file names to train, validation, and test sets based on the split indices
    indices_dict = {
        'train': [file_list[idx] for idx in train_indices],
        'valid': [file_list[idx] for idx in valid_indices],
        'test': [file_list[idx] for idx in test_indices]
    }

    # Save indices to a .pt file
    torch.save(indices_dict, indices_file_path)

    return indices_dict

if __name__ == "__main__":
    
    input_directory = '/content/drive/MyDrive/protein-DATA/sample-final'
    
    indices_dict = stratified_split(input_directory)
