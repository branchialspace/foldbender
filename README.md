# alphafold-transform
Functions for converting Alphafold PDB molecules into graph representations for use with graph networks. Used as data preprocessing for my CAFA 5 project.

# Requirements

```bash
pip install pyg-lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric
sudo apt-get install dssp
pip install rdkit
pip install Bio
pip install ase
pip install dscribe
```
