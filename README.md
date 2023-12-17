# alphafold-transform
Functions for converting Alphafold PDB molecules into graph representations for use with graph networks.

```bash

# Requirements
pip install pyg-lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install torch-geometric
pip install rdkit
pip install Bio
sudo apt-get install dssp
pip install iterative-stratification
pip install ase
pip install dscribe

```



![](alphafold_transform.png)

```bash

# 1
FASTA_ALPHA.fasta_alpha(input_fasta=".fasta", base_directory="")

# 2
ALPHA_NX.alpha_nx(input_directory="", output_directory="")

# 3
NX_PyG.nx_pyg(input_dir="", output_dir="")

# 4
atomic_posenc.soap_local(input_directory="", output_directory="")

# 5 (gpu)
eigen_posenc.precompute_eigens(input_dir="", output_dir="")


# Task specific:

GO_labels.go_labels(input_directory="", output_directory="", train_terms=".tsv")

minmax_norm.minmax_norm(input_dir="", output_dir="")

stratified_split.stratified_split(input_directory="")

ESM2_labels.esm2_labels(embeddings_path=".npy", sequence_ids_path= ".npy", input_dir= "", output_dir= "")
