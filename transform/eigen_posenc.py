# precompute eigenvectors and eigenvalues for graphGPS laplacian positional encodings
import os
import zipfile
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_undirected, get_laplacian
      
def compute_posenc_stats(data, is_undirected):
      """Compute positional encodings for the given graph and store them in the data object.
      
      Args:
        data: PyG graph
        is_undirected: True if the graph is expected to be undirected
      """
      # Basic preprocessing of the input graph.
      N = data.x.shape[0]  # Number of nodes, including disconnected nodes.
      laplacian_norm_type = None
      
      # Filter edges based on the non-zero value in the first position of edge_attr (only if PAE edges are included)
      # mask = data.edge_attr[:, 0].nonzero().view(-1)
      # filtered_edge_index = data.edge_index[:, mask]
      
      if is_undirected:
        undir_edge_index = data.edge_index # or filtered_edge_index
      else:
        undir_edge_index = to_undirected(data.edge_index) # or filtered_edge_index
      
      # Eigen values and vectors.
      evals, evects = None, None
      # Get Laplacian in dense format
      edge_index, edge_weight = get_laplacian(undir_edge_index, normalization=laplacian_norm_type, num_nodes=N)
      edge_index = edge_index.to(device='cuda')
      edge_weight = edge_weight.to(device='cuda')
      
      # Create dense Laplacian matrix
      L = torch.zeros((N, N), device='cuda')
      L[edge_index[0], edge_index[1]] = edge_weight
      
      # Compute eigenvalues and eigenvectors
      max_freqs = 16
      eigvec_norm = "L2"
      evals, evects = torch.linalg.eigh(L)
      EigVals, EigVecs = get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm)
      
      # Store in data object
      data.EigVals = EigVals.to('cpu')
      data.EigVecs = EigVecs.to('cpu')

      return data

def get_lap_decomp_stats(evals, evects, max_freqs, eigvec_norm='L2'):
      """Compute Laplacian eigen-decomposition-based PE stats of the given graph.
      
      Args:
        evals, evects: Precomputed eigen-decomposition
        max_freqs: Maximum number of top smallest frequencies / eigenvecs for padding
        eigvec_norm: Normalization for the eigen vectors of the Laplacian
      Returns:
        Tensor (num_nodes, max_freqs) eigenvalues repeated for each node
        Tensor (num_nodes, max_freqs) of eigenvector values per node
      """
      N = evals.size(0)  # Number of nodes, including disconnected nodes.
      
      # Keep up to the maximum desired number of frequencies.
      idx = evals.argsort()[:max_freqs]
      evals, evects = evals[idx], evects[:, idx]
      evals = evals.clamp_min(0)
      
      # Normalize and pad eigen vectors.
      evects = eigvec_normalizer(evects, evals, normalization=eigvec_norm)
      if N < max_freqs:
        EigVecs = F.pad(evects, (0, max_freqs - N), value=float('nan'))
      else:
        EigVecs = evects
      
      # Pad and save eigenvalues.
      if N < max_freqs:
        EigVals = F.pad(evals, (0, max_freqs - N), value=float('nan')).unsqueeze(0)
      else:
        EigVals = evals.unsqueeze(0)
      EigVals = EigVals.repeat(N, 1)
      
      return EigVals, EigVecs

def eigvec_normalizer(EigVecs, EigVals, normalization="L2", eps=1e-12):
      """
      Implement different eigenvector normalizations.
      """
      
      EigVals = EigVals.unsqueeze(0)
      
      if normalization == "L2":
        # L2 normalization: eigvec / sqrt(sum(eigvec^2))
        denom = EigVecs.norm(p=2, dim=0, keepdim=True)
      
      else:
        raise ValueError(f"Unsupported normalization `{normalization}`")
      
      denom = denom.clamp_min(eps).expand_as(EigVecs)
      EigVecs = EigVecs / denom
      
      return EigVecs

def precompute_eigens(input_dir, output_dir, is_undirected=False, zip_io=True):
    """Process each .pt PyG Data object in the input directory and save to the output directory.
    
    Args:
      input_dir: Directory containing the input .pt PyG Data objects.
      output_dir: Directory where the modified .pt PyG Data objects will be saved.
      sample_size: Number of graphs to sample for determining directedness.
      is_undirected: If False, compute_posenc_stats uses to_undirected
      zip_io: If True, process zipped files.
    """
    # Function to unzip file
    def unzip_file(zip_path, extract_dir):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        # Assuming there is only one file in the zip
        return os.path.join(extract_dir, zip_ref.namelist()[0])

    # Function to zip file
    def zip_file(file_path, zip_path):
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            zip_ref.write(file_path, os.path.basename(file_path))
        os.remove(file_path)

    # File extension to look for
    file_ext = '.pt.zip' if zip_io else '.pt'

    # List all relevant files in the input directory
    dataset_files = [f for f in os.listdir(input_dir) if f.endswith(file_ext)]

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Process each file
    for filename in dataset_files:
        if zip_io:
            # Unzip the file for processing
            data_path = unzip_file(os.path.join(input_dir, filename), input_dir)
        else:
            data_path = os.path.join(input_dir, filename)

        # Load and process the data
        data = torch.load(data_path)
        compute_posenc_stats(data, is_undirected)

        if zip_io:
            # Update filename to .pt for saving
            filename = filename.replace('.zip', '')
            output_path = os.path.join(output_dir, filename)
            torch.save(data, output_path)

            # Zip the processed file
            zip_file(output_path, output_path + '.zip')
        else:
            # Save the processed file directly
            output_path = os.path.join(output_dir, filename)
            torch.save(data, output_path)

if __name__ == "__main__":

    input_dir = "/content/drive/MyDrive/protein-DATA/sample-final"
    output_dir = "/content/drive/MyDrive/protein-DATA/sample_final_eigens"
    
    precompute_eigens(input_dir, output_dir)
