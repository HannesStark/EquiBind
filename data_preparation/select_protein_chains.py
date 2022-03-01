# in this file we perform the removal of chains that have none of their atoms withing a 10 A radius of the ligand.
# this additionally needs "conda install prody"

import os
import warnings

import numpy as np
import prody
from Bio.PDB import PDBIO, PDBParser
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from scipy import spatial
from tqdm import tqdm
from commons.process_mols import read_molecule

cutoff = 10
data_dir = '../data/PDBBind'
names = os.listdir(data_dir)

io = PDBIO()
biopython_parser = PDBParser()
for name in tqdm(names):
    rec_path = os.path.join(data_dir, name, f'{name}_protein_obabel_reduce.pdb')
    lig = read_molecule(os.path.join(data_dir, name, f'{name}_ligand.sdf'), sanitize=True, remove_hs=False)
    if lig == None:
        lig = read_molecule(os.path.join(data_dir, name, f'{name}_ligand.mol2'), sanitize=True, remove_hs=False)
    if lig == None:
        print('ligand was none for ', name)
        with open('select_chains.log', 'a') as file:
            file.write(f'{name}\n')
        continue
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]
    min_distances = []
    coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_is_water = False
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                chain_is_water = True
            residue_coords = []
            c_alpha, n, c = None, None, None
            for atom in residue:
                if atom.name == 'CA':
                    c_alpha = list(atom.get_vector())
                if atom.name == 'N':
                    n = list(atom.get_vector())
                if atom.name == 'C':
                    c = list(atom.get_vector())
                residue_coords.append(list(atom.get_vector()))
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        if len(chain_coords) > 0:
            all_chain_coords = np.concatenate(chain_coords, axis=0)
            distances = spatial.distance.cdist(lig_coords, all_chain_coords)
            min_distance = distances.min()
        else:
            min_distance = np.inf
        if chain_is_water:
            min_distances.append(np.inf)
        else:
            min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        if min_distance < cutoff and not chain_is_water:
            valid_chain_ids.append(chain.get_id())
    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances))
    valid_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())

#  Many thanks to Professor David Ryan Koes for spotting that the commented code only removes water and other chains while keeping the actual receptor chains.
#  While directly modifying the .pdb file as text file is an option, we can again follow Prof. Koes's excellent advice and use the prody library as in the code below.
#    io.set_structure(structure)
#    io.save(os.path.join(data_dir,name,f'{name}_protein_processed2.pdb'))
    prot = prody.parsePDB(rec_path)
    sel = prot.select(' or '.join(map(lambda c: f'chain {c}', valid_chain_ids)))
    prody.writePDB(os.path.join(data_dir,name,f'{name}_protein_processed2.pdb'),sel)