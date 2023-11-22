# Alphafold + RDKit + DSSP > NetworkX Graph Representation of Proteins
import os
import pickle
import json
import networkx as nx
from rdkit import Chem
from Bio.PDB import PDBParser, DSSP

def protein_molecule_graphs(file_name, include_pae=False):
    pdb_file_path = os.path.join(input_directory, file_name + '.pdb')
    json_file_path = os.path.join(input_directory, file_name + '.json')

    output_file_name = file_name.split("/")[-1] + '.pkl'
    output_file_path = os.path.join(output_directory, output_file_name)

    # Check if the pickle file already exists in the output directory
    if os.path.exists(output_file_path):
        print(f"{output_file_name} already exists in the output directory.")
        return

    # Create RDKit molecule from PDB file
    mol = Chem.MolFromPDBFile(pdb_file_path, sanitize=False, removeHs=False)
    mol.UpdatePropertyCache(strict=False)

    # Get Conformer for 3D coordinates
    conf = mol.GetConformer()

    # Create a NetworkX undirected graph
    G = nx.Graph()

    # Parse the PDB file
    pdb_parser = PDBParser()
    structure = pdb_parser.get_structure('protein', pdb_file_path)

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

    # Parse JSON file, Add PAE as Edges only if include_pae is True
    if include_pae:
        try:
            with open(json_file_path, 'r') as f:
                pae_data = json.load(f)

            if not 'predicted_aligned_error' in pae_data[0]:
                raise ValueError('No predicted_aligned_error in JSON')

            pae_matrix = pae_data[0]['predicted_aligned_error']

            for i in range(len(pae_matrix)):
                for j in range(len(pae_matrix[i])):
                    if i != j:  # Skip self-loops
                        pae = pae_matrix[i][j]
                        ca_atom_i = residue_to_ca_atom.get(i + 1)
                        ca_atom_j = residue_to_ca_atom.get(j + 1)

                        if ca_atom_i is not None and ca_atom_j is not None:
                            G.add_edge(ca_atom_i, ca_atom_j, pae=pae)

        except json.JSONDecodeError:
            print(f"Cannot decode JSON from file {file_name}. Please check the JSON file.")
            return
        except ValueError as ve:
            print(f"Value error in file {file_name}: {str(ve)}")
            return
        except Exception as e:
            print(f"Unexpected error in file {file_name}: {str(e)}")
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

    dssp_data = run_dssp(pdb_file_path)

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
    with open(output_file_path, 'wb') as f:
        pickle.dump(G, f)

def process_all_proteins(input_directory, output_directory, include_pae=False):
    processed_files = [f for f in os.listdir(output_directory) if f.endswith('.pkl')]
    for file in os.listdir(input_directory):
        if file.endswith(".pdb"):
            file_name_without_extension = os.path.splitext(file)[0]
            output_file_name = file_name_without_extension + '.pkl'
            output_file_path = os.path.join(output_directory, output_file_name)

            if output_file_name in processed_files:
                continue

            # Process the protein molecule graphs (assuming this function is defined elsewhere)
            protein_molecule_graphs(input_directory, output_directory, file_name_without_extension, include_pae)

            # Add the processed file to the list
            processed_files.append(output_file_name)

input_directory = 'path/to/input_directory'
output_directory = 'path/to/output_directory'

process_all_proteins(input_directory, output_directory)
