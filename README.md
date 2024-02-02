# Foldbender
Functions for converting Alphafold PDB molecules into graph representations for use with graph networks.

```bash

# Requirements (In colab)
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

import alphafold_transform as at

# 1
at.fasta_alpha(input_fasta=".fasta", input_dir="")

# 2
at.alpha_nx(input_dir="", output_dir="")

# 3
at.nx_pyg(input_dir="", output_dir="")

# 4
at.soap_local(input_dir="", r_cut=3, n_max=3, l_max=3, sigma=0.1)

# 5 (gpu)
at.precompute_eigens(input_dir="")


# Task specific:

at.minmax_norm(input_dir="")

at.go_split(input_dir="")

at.go_labels(input_dir="", train_terms=".tsv")

at.esm2_labels(embeddings_path=".npy", sequence_ids_path=".npy", input_dir="")

at.foldseek_targets(file_clusters=".tsv", file_scores=".tsv")

at.foldseek_multiclass_labels(input_dir="", foldseek_targets=".tsv")

at.foldseek_regression_labels(input_dir="", foldseek_targets=".tsv")

at.foldseek_multiclass_split(input_dir="", valid_size=0.3, test_size=0.3, random_state=42)
