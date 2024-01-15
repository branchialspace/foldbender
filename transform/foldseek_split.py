# Multi-label stratified split for Foldseek regression task: Binary bool mask for top n cluster scores
import os
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def foldseek_split(input_directory):
    # Define the path for saving indices
    indices_file_path = os.path.join(os.path.dirname(input_directory), f"{os.path.basename(input_directory)}_split_indices.pt")

    # Load all files in the input directory
    file_list = os.listdir(input_directory)

    # Extract labels for splitting purposes
    label_representations = []
    for file in file_list:
        data = torch.load(os.path.join(input_directory, file))
        # Treat positive values as 1
        label_indices = (data.y > 0).nonzero(as_tuple=True)[0]
        label_representations.append(tuple(sorted(label_indices.tolist())))

    # Convert label representations to a format suitable for MultilabelStratifiedKFold
    unique_labels = sorted(set(sum(label_representations, ())))
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}

    multilabel_format = torch.zeros(len(file_list), len(unique_labels))
    for i, labels in enumerate(label_representations):
        for label in labels:
            multilabel_format[i, label_to_index[label]] = 1

    # First split into 2 folds
    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    first_split_indices = next(iter(mskf.split(multilabel_format, multilabel_format)))

    # Second split on the train set of the first split
    second_split_indices = next(iter(mskf.split(multilabel_format[first_split_indices[0]], multilabel_format[first_split_indices[0]])))

    # Third split on the train set of the second split
    third_split_indices = next(iter(mskf.split(multilabel_format[second_split_indices[0]], multilabel_format[second_split_indices[0]])))

    # Combine validation sets of first 2 splits to create final train indices
    train_indices = torch.cat((torch.tensor(first_split_indices[1]), torch.tensor(second_split_indices[1])))

    # Validation and Test indices
    valid_indices = torch.tensor(third_split_indices[0])
    test_indices = torch.tensor(third_split_indices[1])

    # Assign file names to train, validation, and test sets based on the split indices
    indices_dict = {
        'train': [file_list[idx] for idx in train_indices],
        'valid': [file_list[first_split_indices[0][idx]] for idx in valid_indices],
        'test': [file_list[first_split_indices[0][idx]] for idx in test_indices]
    }

    # Save indices to a .pt file
    torch.save(indices_dict, indices_file_path)

    for key in indices_dict.keys():
        print(f"{key}: {len(indices_dict[key])} files")

if __name__ == "__main__":
    
    input_directory = '/content/drive/MyDrive/protein-DATA/sample-final'
    
    indices_dict = foldseek_split(input_directory)
