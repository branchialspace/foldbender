# Create task_norms file for SingleTaskDataset
import os
import numpy as np
import torch
from torch import Tensor
from graphium.data.normalization import LabelNormalization

base_path = '/datacache/'
subdirs = ['train', 'val', 'test']

# We will store all label data here (both 1 and 0)
all_labels_data = []

# Iterate over subdirectories and process each file
for subdir in subdirs:
    dir_path = os.path.join(base_path, subdir)
    for filename in os.listdir(dir_path):
        if filename.endswith(".pkl") and filename != "multitask_metadata.pkl":
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'rb') as f:
                data = torch.load(f)
                labels = data.get('labels', {})
                # Assuming each file contains all unique GO labels
                label_data = [value.item() for key, value in labels.items() if 'GO:' in key]
                all_labels_data.append(label_data)

# Convert list of label data to numpy array
all_labels_data = np.array(all_labels_data)

# Process arrays and return LabelNormalization object for all labels
label_norm = LabelNormalization(verbose=True)
label_norm.calculate_statistics(all_labels_data.T)  # Transpose to have labels as columns

# If you want to save this to a .pkl file using torch
torch.save({'graph_GO': label_norm}, '/datacache/task_norms.pkl')
