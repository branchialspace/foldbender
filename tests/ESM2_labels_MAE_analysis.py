# ESM2 labels MAE rough analysis
import torch
import os
from torch_geometric.data import Data

directory = '/content/drive/MyDrive/protein-DATA/41k_sample_processed_ESM2/'
mae = 0.0357

# Store all y values
all_y_values = []

for filename in os.listdir(directory):
    if filename.endswith('.pt'):
        file_path = os.path.join(directory, filename)
        data = torch.load(file_path)

        if hasattr(data, 'y'):
            all_y_values.extend(data.y.tolist())

# Convert list back to a tensor
all_y_values = torch.tensor(all_y_values)

# Remove outliers from y
def remove_outliers(tensor, n_std_dev=1.5):
    q1 = torch.quantile(tensor, 0.25)
    q3 = torch.quantile(tensor, 0.75)
    iqr = q3 - q1
    lower_bound = q1 - (n_std_dev * iqr)
    upper_bound = q3 + (n_std_dev * iqr)
    return tensor[(tensor >= lower_bound) & (tensor <= upper_bound)]

cleaned_y_values = remove_outliers(all_y_values)

# Calculate descriptive statistics on cleaned data
mean_val = torch.mean(cleaned_y_values).item()
median_val = torch.median(cleaned_y_values).item()
std_dev = torch.std(cleaned_y_values).item()
min_val = torch.min(cleaned_y_values).item()
max_val = torch.max(cleaned_y_values).item()

print(f"Cleaned Overall Mean of y: {mean_val}")
print(f"Cleaned Overall Median of y: {median_val}")
print(f"Cleaned Overall Standard Deviation of y: {std_dev}")
print(f"Cleaned Overall Minimum value in y: {min_val}")
print(f"Cleaned Overall Maximum value in y: {max_val}")

# Analyzing the MAE in the context of cleaned y's distribution
print(f"Mean Absolute Error (MAE): {mae}")
print(f"MAE as a percentage of the cleaned overall mean value: {mae / mean_val * 100:.2f}%")

# additional stats for comparison
range_y = torch.max(cleaned_y_values) - torch.min(cleaned_y_values)
iqr_y = torch.quantile(cleaned_y_values, 0.75) - torch.quantile(cleaned_y_values, 0.25)

print(f"MAE as a percentage of the range: {mae / range_y.item() * 100:.2f}%")
print(f"MAE as a percentage of the IQR: {mae / iqr_y.item() * 100:.2f}%")
