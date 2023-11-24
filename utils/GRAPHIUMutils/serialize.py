# Serialize
import os
import shutil
from copy import deepcopy
from graphium.config._loader import load_datamodule
from graphium.utils.hashing import get_md5_hash
from graphium.data.datamodule import MultitaskFromSmilesDataModule
from graphium.data.datamodule import DatasetProcessingParams
from omegaconf import OmegaConf

# Modify Hash Function
def get_data_hash_nopop(self):
        args = {}
        for task_key, task_args in deepcopy(self.task_specific_args).items():
            if isinstance(task_args, DatasetProcessingParams):
                task_args = task_args.__dict__  # Convert the class to a dictionary
            # Keep only first 5 rows of a dataframe
            if "df" in task_args.keys():
                if task_args["df"] is not None:
                    task_args["df"] = task_args["df"].iloc[:5]
            args[task_key] = task_args

        hash_dict = {
            "smiles_transformer": self.smiles_transformer,
            "task_specific_args": args,
        }
        data_hash = get_md5_hash(hash_dict)
        return data_hash

MultitaskFromSmilesDataModule.get_data_hash = get_data_hash_nopop

cfg = OmegaConf.to_container(yaml_config, resolve=True)
datamodule_copy = load_datamodule(cfg, "gpu")
data_hash_value = get_data_hash_nopop(datamodule_copy)

# Assign Hash Value
base_path = "/datacache/"

for folder_name in ["train", "val", "test"]:
    original_path = os.path.join(base_path, folder_name)
    new_path = os.path.join(base_path, f"{folder_name}_{data_hash_value}")
    os.rename(original_path, new_path)

new_dir_path = os.path.join(base_path, data_hash_value)
os.makedirs(new_dir_path, exist_ok=True)
original_file_path = os.path.join(base_path, "task_norms.pkl")
new_file_path = os.path.join(new_dir_path, "task_norms.pkl")
shutil.move(original_file_path, new_file_path)

print(data_hash_value)
