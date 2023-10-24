# Multi-label stratified split for train, val, test sets
import os
import torch
import glob
from collections import Counter
from sklearn.model_selection import train_test_split

# Load all .pkl files in the /datacache/ directory
file_list = glob.glob('/datacache/*.pkl')

# Extract labels for splitting purposes
label_representations = []
for file in file_list:
    data = torch.load(file)
    labels = [key for key in data['labels'].keys if key.startswith("GO:")]
    label_representations.append(tuple(sorted(labels)))

# Function to determine if stratification is possible
def can_stratify(labels):
    label_counts = Counter(labels)
    return all(count > 1 for count in label_counts.values())

# If we can stratify based on the labels, we do. Otherwise, we split randomly.
split_args = dict(test_size=0.2, random_state=42)
if can_stratify(label_representations):
    split_args["stratify"] = label_representations

train_files, temp_test_files, _, _ = train_test_split(file_list, label_representations, **split_args)

temp_label_representations = [label_representations[i] for i in range(len(file_list)) if file_list[i] in temp_test_files]
split_args_val_test = dict(test_size=0.5, random_state=42)
if can_stratify(temp_label_representations):
    split_args_val_test["stratify"] = temp_label_representations

val_files, test_files = train_test_split(temp_test_files, **split_args_val_test)

# Function to move files to a directory
def move_files(files, destination_folder):
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)
    for file in files:
        os.rename(file, os.path.join(destination_folder, os.path.basename(file)))

# Move files to their respective directories
move_files(train_files, '/datacache/train')
move_files(val_files, '/datacache/val')
move_files(test_files, '/datacache/test')
