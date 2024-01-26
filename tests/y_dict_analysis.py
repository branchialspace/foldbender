import torch

def check_splits_mutual_exclusivity_and_uniqueness(split_file_path):
    # Load the split dictionary
    split_dict = torch.load(split_file_path)

    # Function to check duplicates within the same split
    def check_duplicates(file_list):
        return len(file_list) != len(set(file_list))

    # Extract file lists from each split and check for internal duplicates
    train_files = split_dict.get('train', [])
    val_files = split_dict.get('val', [])
    test_files = split_dict.get('test', [])

    duplicates_in_train = check_duplicates(train_files)
    duplicates_in_val = check_duplicates(val_files)
    duplicates_in_test = check_duplicates(test_files)

    # Check for mutual exclusivity
    train_val_overlap = set(train_files).intersection(val_files)
    train_test_overlap = set(train_files).intersection(test_files)
    val_test_overlap = set(val_files).intersection(test_files)

    # Test results
    is_mutually_exclusive = not (train_val_overlap or train_test_overlap or val_test_overlap)
    no_internal_duplicates = not (duplicates_in_train or duplicates_in_val or duplicates_in_test)

    print(f"Mutual Exclusivity Test: {'Passed' if is_mutually_exclusive else 'Failed'}")
    print(f"No Internal Duplicates Test: {'Passed' if no_internal_duplicates else 'Failed'}")
    if not is_mutually_exclusive:
        if train_val_overlap:
            print(f"Overlap between train and val: {train_val_overlap}")
        if train_test_overlap:
            print(f"Overlap between train and test: {train_test_overlap}")
        if val_test_overlap:
            print(f"Overlap between val and test: {val_test_overlap}")
    if not no_internal_duplicates:
        if duplicates_in_train:
            print("Duplicates found in train split.")
        if duplicates_in_val:
            print("Duplicates found in val split.")
        if duplicates_in_test:
            print("Duplicates found in test split.")

split_file_path = '/content/drive/MyDrive/protein-DATA/41k_prot_foldseek_split_indices.pt'
check_splits_mutual_exclusivity_and_uniqueness(split_file_path)
