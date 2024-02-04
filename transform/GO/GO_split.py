# Multi-label stratified split for CAFA Gene Ontology Annotations
import os
import torch
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold


def go_split(input_dir):
    # Define the path for saving indices
    indices_file_path = os.path.join(os.path.dirname(input_dir), f"{os.path.basename(input_dir)}_split_indices.pt")

    # Load all files in the input directory
    file_list = os.listdir(input_dir)

    # Load y tensors and stack them
    ys = [torch.load(os.path.join(input_dir, file)).y for file in file_list]
    multilabel_format = torch.stack(ys)

    # First split into 2 folds
    mskf = MultilabelStratifiedKFold(n_splits=2, shuffle=True, random_state=42)
    first_split_indices = next(iter(mskf.split(multilabel_format, multilabel_format)))

    # Second split on the train set of the first split
    second_split_indices = next(iter(mskf.split(multilabel_format[first_split_indices[0]], multilabel_format[first_split_indices[0]])))

    # Adjust second split indices to original indexing
    second_split_indices_adjusted = [first_split_indices[0][idx] for idx in second_split_indices[0]], [first_split_indices[0][idx] for idx in second_split_indices[1]]

    # Third split on the train set of the second split
    third_split_indices = next(iter(mskf.split(multilabel_format[second_split_indices_adjusted[0]], multilabel_format[second_split_indices_adjusted[0]])))

    # Adjust third split indices to original indexing
    third_split_indices_adjusted = [second_split_indices_adjusted[0][idx] for idx in third_split_indices[0]], [second_split_indices_adjusted[0][idx] for idx in third_split_indices[1]]

    # Combine validation sets of first 2 splits to create final train indices
    train_indices = torch.cat((torch.tensor(first_split_indices[1]), torch.tensor(second_split_indices_adjusted[1])))

    # Validation and Test indices
    valid_indices = torch.tensor(third_split_indices_adjusted[0])
    test_indices = torch.tensor(third_split_indices_adjusted[1])

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

if __name__ == "__main__":
    
    input_dir = '/content/drive/MyDrive/protein-DATA/sample-final'
    
    indices_dict = go_split(input_dir)
