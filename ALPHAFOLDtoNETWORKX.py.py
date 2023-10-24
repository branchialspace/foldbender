# Alphafold > NetworkX Graph Representation of Proteins
# Currently operates on GCS
import os
import io
import pickle
import tempfile
import json
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
from Bio.PDB import PDBParser, DSSP

def protein_molecule_graphs(blob_name):
    input_bucket = storage_client.get_bucket(INPUT_BUCKET)
    output_bucket = storage_client.get_bucket(OUTPUT_BUCKET)

    # Check if the pickle file already exists in the output bucket
    output_blob_name = blob_name.split("\\")[-2] + '.pickle'
    if storage.Blob(bucket=output_bucket, name=output_blob_name).exists(storage_client):
        print(f"{output_blob_name} already exists in the output bucket.")
        return

    pdb_blob = input_bucket.blob(blob_name + '.pdb')
    json_blob = input_bucket.blob(blob_name + '.json')

    pdb_text = pdb_blob.download_as_text()
    json_text = json_blob.download_as_text()

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as temp_pdb:
        temp_pdb.write(pdb_text.encode())
        temp_pdb.flush()
        temp_pdb.close()
    pdb_file = temp_pdb.name

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as temp_json:
        temp_json.write(json_text.encode())
        temp_json.flush()
        temp_json.close()
    json_file = temp_json.name

    # Create RDKit molecule from PDB file
    mol = Chem.MolFromPDBFile(pdb_file, sanitize=False, removeHs=False)
    mol.UpdatePropertyCache(strict=False)

    # Get Conformer for 3D coordinates
    conf = mol.GetConformer()

   # Create a NetworkX directed graph
    G = nx.DiGraph()

    # Parse the PDB file
    pdb_parser = PDBParser(QUIET=True)
    structure = pdb_parser.get_structure('protein', pdb_file)

    # Create a dictionary mapping each RDKit atom index to its full atom name
    serial_atom_dict = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    serial_atom_dict[atom.serial_number] = atom.get_fullname().strip()

    # Create a dictionary to store the central atom (alpha carbon) of each residue
    residue_to_ca_atom = {}

    # Iterate through each Atom and add as Nodes, add Atomic Properties as Node Attributes
    for atom in mol.GetAtoms():
        atom_idx = atom.GetIdx()
        atom_name = serial_atom_dict.get(atom.GetMonomerInfo().GetSerialNumber())
        atomic_number = atom.GetAtomicNum()
        atom_coords = conf.GetAtomPosition(atom_idx)
        degree = atom.GetDegree()
        aromatic = atom.GetIsAromatic()
        residue_number = atom.GetMonomerInfo().GetResidueNumber()
        residue_name = atom.GetMonomerInfo().GetResidueName()

        G.add_node(atom_idx, atom_name=atom_name, atomic_number=atomic_number, atom_coords=atom_coords,
                  degree=degree, aromatic=aromatic, residue_number=residue_number, residue_name=residue_name)

        # If this atom is the alpha carbon, store it as the central atom of this residue
        if atom_name == 'CA':
            residue_to_ca_atom[residue_number] = atom_idx

    # Iterate through the Bonds and add as Edges, add Bond Information as Edge Attributes
    for bond in mol.GetBonds():
        atom_i = bond.GetBeginAtom().GetIdx()
        atom_j = bond.GetEndAtom().GetIdx()
        bond_idx = bond.GetIdx()
        bond_order = bond.GetBondTypeAsDouble()
        bond_length = rdMolTransforms.GetBondLength(conf, atom_i, atom_j)

        G.add_edge(atom_i, atom_j, bond_idx=bond_idx, bond_order=bond_order, bond_length=bond_length)

    # Identify pLDDT as Node Attributes and PAE as Edges

    # Parse the PDB file
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure('protein', pdb_file)

    # Create a dictionary mapping each residue to its pLDDT value
    plddt_dict = {}
    for model in structure:
        for chain in model:
            for residue in chain:
                # Note: assumes that the pLDDT is stored in the B-factor field of the alpha carbon atom
                plddt = residue['CA'].get_bfactor()
                plddt_dict[residue.get_id()[1]] = plddt

    # Add pLDDT as Node Attributes
    for atom in mol.GetAtoms():
        residue_number = atom.GetMonomerInfo().GetResidueNumber()
        plddt = plddt_dict.get(residue_number)
        G.nodes[atom.GetIdx()]['plddt'] = plddt

    # Parse JSON file, Add PAE as Edges
    try:
        with open(json_file, 'r') as f:
            pae_data = json.load(f)

        # Check if 'predicted_aligned_error' is present in the data
        if not 'predicted_aligned_error' in pae_data[0]:
            raise ValueError('No predicted_aligned_error in JSON')

        pae_matrix = pae_data[0]['predicted_aligned_error']

        for i in range(len(pae_matrix)):
            for j in range(len(pae_matrix[i])):
                if i != j:  # Skip self-loops
                    pae = pae_matrix[i][j]

                    # Get the central atoms of the residues directly from the mapping
                    ca_atom_i = residue_to_ca_atom.get(i + 1)
                    ca_atom_j = residue_to_ca_atom.get(j + 1)

                    # Add an Edge between the central atoms with the PAE as an attribute
                    G.add_edge(ca_atom_i, ca_atom_j, pae=pae)

    except json.JSONDecodeError:
        print(f"Cannot decode JSON from blob {blob_name}. Please check the JSON file.")
        return
    except ValueError as ve:
        print(f"Value error in blob {blob_name}: {str(ve)}")
        return
    except Exception as e:
        print(f"Unexpected error in blob {blob_name}: {str(e)}")
        return

    # Identify DSSP Secondary Structures, Solvent Available Surface Area, Torsion Angles, Hygrogen Bond Strengths. Map the DSSP data to residue identifiers as Node Attributes
    def run_dssp(pdb_file):
        p = PDBParser()
        s = p.get_structure("protein", pdb_file)
        model = s[0]
        dssp = DSSP(model, pdb_file)

        # Convert the DSSP output to match the graph nodes format
        dssp_dict = {}
        for res in dssp:
            dssp_dict[res[0]] = res

        return dssp_dict

    dssp_data = run_dssp(pdb_file)

    for node, data in G.nodes(data=True):
        if data['residue_number'] in dssp_data:
            dssp_node_data = dssp_data[data['residue_number']]

            # Unpack DSSP data
            (dssp_index, aa, ss, exposure, phi, psi, NH_O_1_relidx, NH_O_1_energy, O_NH_1_relidx, O_NH_1_energy,
            NH_O_2_relidx, NH_O_2_energy, O_NH_2_relidx, O_NH_2_energy) = dssp_node_data

            # Update node attributes
            G.nodes[node]['secondary_structure'] = ss
            G.nodes[node]['exposure'] = exposure
            G.nodes[node]['phi'] = phi
            G.nodes[node]['psi'] = psi
            G.nodes[node]['NH_O_1_relidx'] = NH_O_1_relidx
            G.nodes[node]['NH_O_1_energy'] = NH_O_1_energy
            G.nodes[node]['O_NH_1_relidx'] = O_NH_1_relidx
            G.nodes[node]['O_NH_1_energy'] = O_NH_1_energy
            G.nodes[node]['NH_O_2_relidx'] = NH_O_2_relidx
            G.nodes[node]['NH_O_2_energy'] = NH_O_2_energy
            G.nodes[node]['O_NH_2_relidx'] = O_NH_2_relidx
            G.nodes[node]['O_NH_2_energy'] = O_NH_2_energy

    # Convert atom_coords to string
    for node, data in G.nodes(data=True):
        atom_coords = data['atom_coords']
        atom_coords_str = f"{atom_coords.x},{atom_coords.y},{atom_coords.z}"
        data['atom_coords'] = atom_coords_str

    # Save graph to pickle file
    with tempfile.NamedTemporaryFile(suffix=".pickle", delete=False) as temp_pickle:
        pickle.dump(G, temp_pickle)
        temp_pickle.close()

    # Upload the pickle file to the bucket
    pickle_blob = output_bucket.blob(output_blob_name)
    pickle_blob.upload_from_filename(temp_pickle.name)
    os.remove(temp_pickle.name)

def get_last_processed_blob_name():
    try:
        with open(LAST_BLOB, "r") as file:
            return file.read().strip()
    except FileNotFoundError:
        return None

def set_last_processed_blob_name(blob_name):
    with open(LAST_BLOB, "w") as file:
        file.write(blob_name)

# Iterate over all PDB files in the Input Bucket
input_bucket = storage_client.get_bucket(INPUT_BUCKET)
last_processed_blob_name = get_last_processed_blob_name()
resume_processing = False if last_processed_blob_name is None else True

for blob in input_bucket.list_blobs():
    if blob.name.endswith(".pdb"):
        # Skip blobs until we reach the last processed blob
        if resume_processing:
            if blob.name[:-4] == last_processed_blob_name:
                # We found the last processed blob, resume from the next blob
                resume_processing = False
            continue

        # Process this blob
        protein_molecule_graphs(blob.name[:-4])

        # Update the last processed blob
        set_last_processed_blob_name(blob.name[:-4])
