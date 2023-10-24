# Create Batches from Splits
import os
import shutil
import torch
from graphium.data.dataset import MultitaskDataset

root_dir = "/datacache/"
batch_size = 1

# Define the prefix patterns
prefixes = ["train", "val", "test"]

for prefix in prefixes:
    subdirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d)) and d.startswith(prefix)]

    for subdir in subdirs:
        subdir_path = os.path.join(root_dir, subdir)
        files = sorted([f for f in os.listdir(subdir_path) if f.endswith(".pkl") and f != "multitask_metadata.pkl"])

        total_files = len(files)

        for idx, file in enumerate(files):
            # Calculate the new directory and file name based on the index
            new_dir = os.path.join(subdir_path, format(idx // batch_size, "04d"))
            new_file = os.path.join(new_dir, format(idx, "07d") + ".pkl")

            # Ensure the new directory exists
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # Move the file to its new location
            shutil.move(os.path.join(subdir_path, file), new_file)

        print(f"Processed {total_files} files in {subdir}")

    def load_graph_from_index_batch(self, data_idx):
        filename = os.path.join(
            self.data_path, format(data_idx // batch_size, "04d"), format(data_idx, "07d") + ".pkl"
        )
        data_dict = torch.load(filename)
        return data_dict

MultitaskDataset.load_graph_from_index = load_graph_from_index_batch
