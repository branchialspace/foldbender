import os
import torch
from torch_geometric.data import Data

def analyze_pyg_data(directory_path):
    count_all_zeros = 0
    count_no_y_or_diff_shape = 0
    y_shape = None

    for file in os.listdir(directory_path):
        if file.endswith('.pt'):
            file_path = os.path.join(directory_path, file)
            data = torch.load(file_path)

            if isinstance(data, Data) and hasattr(data, 'y'):
                # Check if y tensor shape is consistent
                if y_shape is None:
                    y_shape = data.y.shape
                elif data.y.shape != y_shape:
                    count_no_y_or_diff_shape += 1
                    continue

                # Check if all values in y tensor are zeros
                if torch.all(data.y == 0):
                    count_all_zeros += 1
            else:
                count_no_y_or_diff_shape += 1

    return count_all_zeros, count_no_y_or_diff_shape

# Example usage
directory_path = '/content/41k_prot_go_1'
count_zeros, count_no_y_or_diff_shape = analyze_pyg_data(directory_path)
print(f"Number of .pt files where all y values are zeros: {count_zeros}")
print(f"Number of .pt files with no y tensor or y tensor of different shape: {count_no_y_or_diff_shape}")
