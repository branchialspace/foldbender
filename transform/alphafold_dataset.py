import os
import re
import torch
from torch_geometric.data import Dataset

class AlphafoldDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(AlphafoldDataset, self).__init__(root, transform, pre_transform)
        self.data_files = sorted(
            [f for f in os.listdir(root) if f.endswith('.pt')],
            key=lambda x: int(re.findall(r'\d+', x)[0])
        )
        self.split_dict = torch.load(os.path.join(root, 'split_dict.pt'))
        
    def get_idx_split(self):
        return self.split_dict
