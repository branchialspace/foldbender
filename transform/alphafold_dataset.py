from typing import List
import os
import os.path as osp
import re
import torch
from torch_geometric.data import Dataset, Data

class Alphafold(Dataset):
    def __init__(self):
        self.root = '/content/drive/MyDrive/protein-DATA/sample-final'
        self.split_file = '/content/drive/MyDrive/protein-DATA/dataset-indices.pt'

    @property
    def processed_file_names(self) -> List[str]:
        return sorted(
            [f for f in os.listdir(self.root) if f.endswith('.pt')],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data_path = osp.join(self.root, self.processed_file_names[idx])
        data = torch.load(data_path)
        return data

    def get_idx_split(self):
        split_dict = torch.load(self.split_file)
        return split_dict

    @property
    def data(self):
        return self.processed_file_names
