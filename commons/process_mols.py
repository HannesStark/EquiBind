import math
import warnings

import pandas as pd
import dgl
import numpy as np
import scipy.spatial as spa
import torch
from Bio.PDB import get_surface, PDBParser, ShrakeRupley
from Bio.PDB.PDBExceptions import PDBConstructionWarning
from biopandas.pdb import PandasPdb
from rdkit import Chem
from rdkit.Chem import MolFromPDBFile, AllChem, GetPeriodicTable, rdDistGeom
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from scipy import spatial
from scipy.special import softmax

from commons.geometry_utils import rigid_transform_Kabsch_3D, rigid_transform_Kabsch_3D_torch
from commons.logger import log

biopython_parser = PDBParser()
periodic_table = GetPeriodicTable()
allowable_features = {
    'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
    'possible_chirality_list': [
        'CHI_UNSPECIFIED',
        'CHI_TETRAHEDRAL_CW',
        'CHI_TETRAHEDRAL_CCW',
        'CHI_OTHER'
    ],
    'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
    'possible_numring_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_implicit_valence_list': [0, 1, 2, 3, 4, 5, 6, 'misc'],
    'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
    'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
    'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
    'possible_hybridization_list': [
        'SP', 'SP2', 'SP3', 'SP3D', 'SP3D2', 'misc'
    ],
    'possible_is_aromatic_list': [False, True],
    'possible_is_in_ring3_list': [False, True],
    'possible_is_in_ring4_list': [False, True],
    'possible_is_in_ring5_list': [False, True],
    'possible_is_in_ring6_list': [False, True],
    'possible_is_in_ring7_list': [False, True],
    'possible_is_in_ring8_list': [False, True],
    'possible_amino_acids': ['ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET',
                             'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL', 'HIP', 'HIE', 'TPO', 'HID', 'LEV', 'MEU',
                             'PTR', 'GLV', 'CYT', 'SEP', 'HIZ', 'CYM', 'GLM', 'ASQ', 'TYS', 'CYX', 'GLZ', 'misc'],
    'possible_atom_type_2': ['C*', 'CA', 'CB', 'CD', 'CE', 'CG', 'CH', 'CZ', 'N*', 'ND', 'NE', 'NH', 'NZ', 'O*', 'OD',
                             'OE', 'OG', 'OH', 'OX', 'S*', 'SD', 'SG', 'misc'],
    'possible_atom_type_3': ['C', 'CA', 'CB', 'CD', 'CD1', 'CD2', 'CE', 'CE1', 'CE2', 'CE3', 'CG', 'CG1', 'CG2', 'CH2',
                             'CZ', 'CZ2', 'CZ3', 'N', 'ND1', 'ND2', 'NE', 'NE1', 'NE2', 'NH1', 'NH2', 'NZ', 'O', 'OD1',
                             'OD2', 'OE1', 'OE2', 'OG', 'OG1', 'OH', 'OXT', 'SD', 'SG', 'misc'],
}

lig_feature_dims = (list(map(len, [
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_chirality_list'],
    allowable_features['possible_degree_list'],
    allowable_features['possible_formal_charge_list'],
    allowable_features['possible_implicit_valence_list'],
    allowable_features['possible_numH_list'],
    allowable_features['possible_number_radical_e_list'],
    allowable_features['possible_hybridization_list'],
    allowable_features['possible_is_aromatic_list'],
    allowable_features['possible_numring_list'],
    allowable_features['possible_is_in_ring3_list'],
    allowable_features['possible_is_in_ring4_list'],
    allowable_features['possible_is_in_ring5_list'],
    allowable_features['possible_is_in_ring6_list'],
    allowable_features['possible_is_in_ring7_list'],
    allowable_features['possible_is_in_ring8_list'],
])), 1)  # number of scalar features
rec_atom_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids'],
    allowable_features['possible_atomic_num_list'],
    allowable_features['possible_atom_type_2'],
    allowable_features['possible_atom_type_3'],
])), 2)

rec_residue_feature_dims = (list(map(len, [
    allowable_features['possible_amino_acids']
])), 2)


def lig_atom_featurizer(mol):
    ComputeGasteigerCharges(mol)  # they are Nan for 93 molecules in all of PDBbind. We put a 0 in that case.
    ringinfo = mol.GetRingInfo()
    atom_features_list = []
    for idx, atom in enumerate(mol.GetAtoms()):
        g_charge = atom.GetDoubleProp('_GasteigerCharge')
        atom_features_list.append([
            safe_index(allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
            allowable_features['possible_chirality_list'].index(str(atom.GetChiralTag())),
            safe_index(allowable_features['possible_degree_list'], atom.GetTotalDegree()),
            safe_index(allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
            safe_index(allowable_features['possible_implicit_valence_list'], atom.GetImplicitValence()),
            safe_index(allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
            safe_index(allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
            safe_index(allowable_features['possible_hybridization_list'], str(atom.GetHybridization())),
            allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
            safe_index(allowable_features['possible_numring_list'], ringinfo.NumAtomRings(idx)),
            allowable_features['possible_is_in_ring3_list'].index(ringinfo.IsAtomInRingOfSize(idx, 3)),
            allowable_features['possible_is_in_ring4_list'].index(ringinfo.IsAtomInRingOfSize(idx, 4)),
            allowable_features['possible_is_in_ring5_list'].index(ringinfo.IsAtomInRingOfSize(idx, 5)),
            allowable_features['possible_is_in_ring6_list'].index(ringinfo.IsAtomInRingOfSize(idx, 6)),
            allowable_features['possible_is_in_ring7_list'].index(ringinfo.IsAtomInRingOfSize(idx, 7)),
            allowable_features['possible_is_in_ring8_list'].index(ringinfo.IsAtomInRingOfSize(idx, 8)),
            g_charge if not np.isnan(g_charge) and not np.isinf(g_charge) else 0.
        ])

    return torch.tensor(atom_features_list)


sr = ShrakeRupley(probe_radius=1.4,  # in A. Default is 1.40 roughly the radius of a water molecule.
                  n_points=100)  # resolution of the surface of each atom. Default is 100. A higher number of points results in more precise measurements, but slows down the calculation.

def rec_atom_featurizer(rec, surface_indices):
    surface_atom_feat = []
    c_alpha_feat = []
    sr.compute(rec, level="A")
    for i, atom in enumerate(rec.get_atoms()):
        if i in surface_indices or atom.name == 'CA':
            atom_name, element = atom.name, atom.element
            sasa = atom.sasa
            bfactor = atom.bfactor
            if element == 'CD':
                element = 'C'
            assert not element == ''
            assert not np.isinf(bfactor)
            assert not np.isnan(bfactor)
            assert not np.isinf(sasa)
            assert not np.isnan(sasa)
            try:
                atomic_num = periodic_table.GetAtomicNumber(element)
            except:
                atomic_num = -1
            atom_feat = [safe_index(allowable_features['possible_amino_acids'], atom.get_parent().get_resname()),
                         safe_index(allowable_features['possible_atomic_num_list'], atomic_num),
                         safe_index(allowable_features['possible_atom_type_2'], (atom_name + '*')[:2]),
                         safe_index(allowable_features['possible_atom_type_3'], atom_name),
                         sasa,
                         bfactor]
            if i in surface_indices:
                surface_atom_feat.append(atom_feat)
            if atom.name == 'CA':
                c_alpha_feat.append(atom_feat)
    return torch.tensor(c_alpha_feat, dtype=torch.float32), torch.tensor(surface_atom_feat, dtype=torch.float32)

def get_receptor_atom_subgraph(rec, rec_coords, lig,  lig_coords=None ,graph_cutoff=4, max_neighbor=8, subgraph_radius=7):
    lig_coords = lig.GetConformer().GetPositions() if lig_coords == None else lig_coords
    rec_coords = np.concatenate(rec_coords, axis=0)
    sr.compute(rec, level="A")
    lig_rec_distance = spa.distance.cdist(lig_coords, rec_coords)
    subgraph_indices = np.where(np.min(lig_rec_distance, axis=0) < subgraph_radius)[0]
    subgraph_coords = rec_coords[subgraph_indices]
    distances = spa.distance.cdist(subgraph_coords, subgraph_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(len(subgraph_coords)):
        dst = list(np.where(distances[i, :] < graph_cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            log(f'The graph_cutoff {graph_cutoff} was too small for one c_alpha such that it had no neighbors. So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = subgraph_coords[src, :] - subgraph_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=len(subgraph_coords), idtype=torch.int32)

    _, features = rec_atom_featurizer(rec, surface_indices=list(subgraph_indices))
    graph.ndata['feat'] = features
    graph.edata['feat'] = distance_featurizer(dist_list, divisor=1)  # avg distance = 7. So divisor = (4/7)*7 = 4
    graph.ndata['x'] = torch.from_numpy(subgraph_coords.astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    log('number of subgraph nodes = ', len(subgraph_coords), ' number of edges in subgraph = ', len(dist_list) )
    return graph

def rec_residue_featurizer(rec):
    feature_list = []
    sr.compute(rec, level="R")
    for residue in rec.get_residues():
        sasa = residue.sasa
        for atom in residue:
            if atom.name == 'CA':
                bfactor = atom.bfactor
        assert not np.isinf(bfactor)
        assert not np.isnan(bfactor)
        assert not np.isinf(sasa)
        assert not np.isnan(sasa)
        feature_list.append([safe_index(allowable_features['possible_amino_acids'], residue.get_resname()),
                             sasa,
                             bfactor])
    return torch.tensor(feature_list, dtype=torch.float32)  # (N_res, 1)


class AtomEncoder(torch.nn.Module):

    def __init__(self, emb_dim, feature_dims, use_scalar_feat=True, n_feats_to_use=None):
        # first element of feature_dims tuple is a list with the lenght of each categorical feature and the second is the number of scalar features
        super(AtomEncoder, self).__init__()
        self.use_scalar_feat = use_scalar_feat
        self.n_feats_to_use = n_feats_to_use
        self.atom_embedding_list = torch.nn.ModuleList()
        self.num_categorical_features = len(feature_dims[0])
        self.num_scalar_features = feature_dims[1]
        for i, dim in enumerate(feature_dims[0]):
            emb = torch.nn.Embedding(dim, emb_dim)
            torch.nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0:
            self.linear = torch.nn.Linear(self.num_scalar_features, emb_dim)

    def forward(self, x):
        x_embedding = 0
        assert x.shape[1] == self.num_categorical_features + self.num_scalar_features
        for i in range(self.num_categorical_features):
            x_embedding += self.atom_embedding_list[i](x[:, i].long())
            if i + 1 == self.n_feats_to_use:
                break

        if self.num_scalar_features > 0 and self.use_scalar_feat:
            x_embedding += self.linear(x[:, self.num_categorical_features:])
        if torch.isnan(x_embedding).any():
            log('nan')
        return x_embedding


def safe_index(l, e):
    """
    Return index of element e in list l. If e is not present, return the last index
    """
    try:
        return l.index(e)
    except:
        return len(l) - 1

def get_receptor(rec_path, lig, cutoff):
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]
    min_distances = []
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
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
            # TODO: Also include the chain_coords.append(np.array(residue_coords)) for non amino acids such that they can be used when using the atom representation of the receptor
            if c_alpha != None and n != None and c != None:  # only append residue if it is an amino acid and not some weired molecule that is part of the complex
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
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

        min_distances.append(min_distance)
        lengths.append(count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        if min_distance < cutoff:
            valid_chain_ids.append(chain.get_id())
    min_distances = np.array(min_distances)
    if len(valid_chain_ids) == 0:
        valid_chain_ids.append(np.argmin(min_distances))
    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]

    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    return rec, coords, c_alpha_coords, n_coords, c_coords

def get_receptor_inference(rec_path):
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=PDBConstructionWarning)
        structure = biopython_parser.get_structure('random_id', rec_path)
        rec = structure[0]
    coords = []
    c_alpha_coords = []
    n_coords = []
    c_coords = []
    valid_chain_ids = []
    lengths = []
    for i, chain in enumerate(rec):
        chain_coords = []  # num_residues, num_atoms, 3
        chain_c_alpha_coords = []
        chain_n_coords = []
        chain_c_coords = []
        count = 0
        invalid_res_ids = []
        for res_idx, residue in enumerate(chain):
            if residue.get_resname() == 'HOH':
                invalid_res_ids.append(residue.get_id())
                continue
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
                chain_c_alpha_coords.append(c_alpha)
                chain_n_coords.append(n)
                chain_c_coords.append(c)
                chain_coords.append(np.array(residue_coords))
                count += 1
            else:
                invalid_res_ids.append(residue.get_id())
        for res_id in invalid_res_ids:
            chain.detach_child(res_id)
        lengths.append(count)
        coords.append(chain_coords)
        c_alpha_coords.append(np.array(chain_c_alpha_coords))
        n_coords.append(np.array(chain_n_coords))
        c_coords.append(np.array(chain_c_coords))
        if len(chain_coords) > 0:
            valid_chain_ids.append(chain.get_id())
    valid_coords = []
    valid_c_alpha_coords = []
    valid_n_coords = []
    valid_c_coords = []
    valid_lengths = []
    invalid_chain_ids = []
    for i, chain in enumerate(rec):
        if chain.get_id() in valid_chain_ids:
            valid_coords.append(coords[i])
            valid_c_alpha_coords.append(c_alpha_coords[i])
            valid_n_coords.append(n_coords[i])
            valid_c_coords.append(c_coords[i])
            valid_lengths.append(lengths[i])
        else:
            invalid_chain_ids.append(chain.get_id())
    coords = [item for sublist in valid_coords for item in sublist]  # list with n_residues arrays: [n_atoms, 3]

    c_alpha_coords = np.concatenate(valid_c_alpha_coords, axis=0)  # [n_residues, 3]
    n_coords = np.concatenate(valid_n_coords, axis=0)  # [n_residues, 3]
    c_coords = np.concatenate(valid_c_coords, axis=0)  # [n_residues, 3]

    for invalid_id in invalid_chain_ids:
        rec.detach_child(invalid_id)

    assert len(c_alpha_coords) == len(n_coords)
    assert len(c_alpha_coords) == len(c_coords)
    assert sum(valid_lengths) == len(c_alpha_coords)
    return rec, coords, c_alpha_coords, n_coords, c_coords

def get_rdkit_coords(mol, seed = None):
    ps = AllChem.ETKDGv2()
    if seed is not None:
        ps.randomSeed = seed
    id = AllChem.EmbedMolecule(mol, ps)
    if id == -1:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        AllChem.EmbedMolecule(mol, ps)
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    else:
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
    conf = mol.GetConformer()
    lig_coords = conf.GetPositions()
    return torch.tensor(lig_coords, dtype=torch.float32)

def get_multiple_rdkit_coords(mol,num_conf=10):
    ps = AllChem.ETKDGv2()
    ids = rdDistGeom.EmbedMultipleConfs(mol, num_conf, ps)
    if -1 in ids:
        print('rdkit coords could not be generated without using random coords. using random coords now.')
        ps.useRandomCoords = True
        ids = rdDistGeom.EmbedMultipleConfs(mol, num_conf, ps)
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    else:
        AllChem.MMFFOptimizeMoleculeConfs(mol)
    conformers = []
    for i in range(num_conf):
        conformers.append(mol.GetConformer(i).GetPositions())

    return np.array(conformers)

def get_multiple_rdkit_coords_individual(mol,num_conf=10):
    conformers = []
    attempts = 0
    while len(conformers) != num_conf:
        try:
            ps = AllChem.ETKDGv2()
            id = AllChem.EmbedMolecule(mol, ps)
            if id == -1:
                print('rdkit coords could not be generated without using random coords. using random coords now.')
                ps.useRandomCoords = True
                AllChem.EmbedMolecule(mol, ps)
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
            else:
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
            conformers.append(mol.GetConformer().GetPositions())
        except Exception as e:
            if attempts == 230: raise Exception(e)
            attempts+= 1
    return np.array(conformers)

def get_pocket_coords(lig, rec_coords, cutoff=5.0, pocket_mode='match_atoms'):
    rec_coords = np.concatenate(rec_coords, axis=0)
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    if pocket_mode == 'match_atoms':
        lig_rec_distance = spa.distance.cdist(lig_coords, rec_coords)
        closest_rec_idx = np.argmin(lig_rec_distance, axis=1)
        satisfies_cufoff_mask = np.where(np.min(lig_rec_distance, axis=1) < cutoff)
        rec_pocket_coords = rec_coords[closest_rec_idx[satisfies_cufoff_mask]]
        lig_pocket_coords = lig_coords[satisfies_cufoff_mask]
        pocket_coords = 0.5 * (lig_pocket_coords + rec_pocket_coords)
    elif pocket_mode == 'lig_atoms':
        pocket_coords = lig_coords
    elif pocket_mode == 'match_atoms_to_lig':
        lig_rec_distance = spa.distance.cdist(lig_coords, rec_coords)
        satisfies_cufoff_mask = np.where(np.min(lig_rec_distance, axis=1) < cutoff)
        lig_pocket_coords = lig_coords[satisfies_cufoff_mask]
        pocket_coords = lig_pocket_coords
    elif pocket_mode == 'match_terminal_atoms':
        terminal_idx = []
        for i, atom in enumerate(lig.GetAtoms()):
            if atom.GetDegree() <= 1:
                terminal_idx.append(i)
        terminal_coords = lig_coords[terminal_idx]
        lig_rec_distance = spa.distance.cdist(terminal_coords, rec_coords)
        closest_rec_idx = np.argmin(lig_rec_distance, axis=1)
        satisfies_cufoff_mask = np.where(np.min(lig_rec_distance, axis=1) < cutoff)
        rec_pocket_coords = rec_coords[closest_rec_idx[satisfies_cufoff_mask]]
        lig_pocket_coords = lig_coords[np.array(terminal_idx)[satisfies_cufoff_mask]]
        pocket_coords = 0.5 * (lig_pocket_coords + rec_pocket_coords)
    elif pocket_mode == 'radius_based':
        # Keep pairs of lig and rec residues/atoms that have pairwise distances < threshold
        lig_rec_distance = spa.distance.cdist(lig_coords, rec_coords)
        positive_tuple = np.where(lig_rec_distance < cutoff)
        active_lig = positive_tuple[0]
        active_rec = positive_tuple[1]
        dynamic_cutoff = cutoff
        while active_lig.size < 4:
            log(
                'Increasing pocket cutoff radius by 0.5 because there were less than 4 pocket nodes with radius: ',
                dynamic_cutoff)
            dynamic_cutoff += 0.5
            positive_tuple = np.where(lig_rec_distance < dynamic_cutoff)
            active_lig = positive_tuple[0]
            active_rec = positive_tuple[1]
        # pos_idx = np.stack([active_lig, active_rec], axis=1)
        lig_pocket_coords = lig_coords[active_lig, :]
        rec_pocket_coords = rec_coords[active_rec, :]
        assert np.max(np.linalg.norm(lig_pocket_coords - rec_pocket_coords, axis=1)) <= dynamic_cutoff
        pocket_coords = 0.5 * (lig_pocket_coords + rec_pocket_coords)
    else:
        raise ValueError(f'pocket_mode -{pocket_mode}- not supported')
    log('Num pocket nodes = ', len(pocket_coords), 'ligand nodes = ', lig_coords.shape[0],
        'receptor num all atoms = ',
        rec_coords.shape[0])
    return torch.tensor(pocket_coords, dtype=torch.float32)


def complex_to_graph(lig, rec, rec_coords, c_alpha_coords, n_coords, c_coords, use_rec_atoms, lig_radius, rec_radius,
                     surface_graph_cutoff, surface_mesh_cutoff, c_alpha_max_neighbors=None, surface_max_neighbors=None,
                     lig_max_neighbors=None):
    conf = lig.GetConformer()
    lig_coords = conf.GetPositions()
    lig_graph = get_lig_graph(lig, lig_coords, lig_radius, max_neighbor=lig_max_neighbors)
    if use_rec_atoms:
        rec_graph = get_hierarchical_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords,
                                           c_alpha_cutoff=rec_radius,
                                           surface_mesh_cutoff=surface_mesh_cutoff,
                                           c_alpha_max_neighbors=c_alpha_max_neighbors,
                                           surface_max_neighbors=surface_max_neighbors,
                                           surface_graph_cutoff=surface_graph_cutoff,
                                           )
    else:
        rec_graph = get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, rec_radius, c_alpha_max_neighbors)
    complex_graph = lig_rec_graphs_to_complex_graph(lig_graph, rec_graph)
    return complex_graph

def get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords, use_rec_atoms, rec_radius,
                     surface_graph_cutoff, surface_mesh_cutoff, c_alpha_max_neighbors=None, surface_max_neighbors=None):
    if use_rec_atoms:
        return get_hierarchical_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords,
                                           c_alpha_cutoff=rec_radius,
                                           surface_mesh_cutoff=surface_mesh_cutoff,
                                           c_alpha_max_neighbors=c_alpha_max_neighbors,
                                           surface_max_neighbors=surface_max_neighbors,
                                           surface_graph_cutoff=surface_graph_cutoff,
                                           )
    else:
        return get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, rec_radius, c_alpha_max_neighbors)

def get_lig_graph(mol, lig_coords, radius=20, max_neighbor=None):
    ################### Build the k-NN graph ##############################
    num_nodes = lig_coords.shape[0]
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)

    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbor + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            log(
                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = lig_coords[src, :] - lig_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)

        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(mol)
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph

def get_lig_structure_graph(lig):
    coords = lig.GetConformer().GetPositions()
    weights = []
    for idx, atom in enumerate(lig.GetAtoms()):
        weights.append(atom.GetAtomicNum())
    weights = np.array(weights)
    mask = []
    angles = []
    edges = []
    distances = []
    for bond in lig.GetBonds():
        type = bond.GetBondType()
        src_idx = bond.GetBeginAtomIdx()
        dst_idx = bond.GetEndAtomIdx()
        src = lig.GetAtomWithIdx(src_idx)
        dst = lig.GetAtomWithIdx(dst_idx)
        src_neighbors = [atom.GetIdx() for atom in list(src.GetNeighbors())]
        src_neighbors.remove(dst_idx)
        src_weights = weights[src_neighbors]
        dst_neighbors = [atom.GetIdx() for atom in list(dst.GetNeighbors())]
        dst_neighbors.remove(src_idx)
        dst_weights = weights[dst_neighbors]
        src_to_dst = (coords[dst_idx] - coords[src_idx])
        if not (len(src_neighbors) > 0 and len(
                dst_neighbors) > 0) or type != Chem.rdchem.BondType.SINGLE or bond.IsInRing():
            edges.append([src_idx, dst_idx])
            distances.append(np.linalg.norm(src_to_dst))
            mask.append(0)
            angles.append(-1)
            edges.append([dst_idx, src_idx])
            distances.append(np.linalg.norm(src_to_dst))
            mask.append(0)
            angles.append(-1)
            continue
        src_neighbor_coords = coords[src_neighbors]
        dst_neighbor_coords = coords[dst_neighbors]
        src_mean_vec = np.mean(src_neighbor_coords * np.array(src_weights)[:, None] - coords[src_idx], axis=0)
        dst_mean_vec = np.mean(dst_neighbor_coords * np.array(dst_weights)[:, None] - coords[dst_idx], axis=0)
        normal = src_to_dst / np.linalg.norm(src_to_dst)
        src_mean_projection = src_mean_vec - src_mean_vec.dot(normal) * normal
        dst_mean_projection = dst_mean_vec - dst_mean_vec.dot(normal) * normal
        cos_dihedral = src_mean_projection.dot(dst_mean_projection) / (
                    np.linalg.norm(src_mean_projection) * np.linalg.norm(dst_mean_projection))
        dihedral_angle = np.arccos(cos_dihedral)
        edges.append([src_idx, dst_idx])
        mask.append(1)
        distances.append(np.linalg.norm(src_to_dst))
        angles.append(dihedral_angle)
        edges.append([dst_idx, src_idx])
        distances.append(np.linalg.norm(src_to_dst))
        mask.append(1)
        angles.append(dihedral_angle)
    edges = torch.tensor(edges)
    graph = dgl.graph((edges[:, 0], edges[:, 1]), num_nodes=len(coords), idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(lig)
    graph.ndata['weights'] = torch.from_numpy(np.array(weights).astype(np.float32))
    graph.edata['feat'] = distance_featurizer(distances, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(coords).astype(np.float32))

    return graph, torch.tensor(mask, dtype=bool), torch.tensor(angles, dtype=torch.float32)

def get_geometry_graph(lig):
    coords = lig.GetConformer().GetPositions()
    edges_src = []
    edges_dst = []
    for i, atom in enumerate(lig.GetAtoms()):
        src_idx = atom.GetIdx()
        assert src_idx == i
        one_hop_dsts = [neighbor for neighbor in list(atom.GetNeighbors())]
        two_and_one_hop_idx = [neighbor.GetIdx() for neighbor in one_hop_dsts]
        for one_hop_dst in one_hop_dsts:
            for two_hop_dst in one_hop_dst.GetNeighbors():
                two_and_one_hop_idx.append(two_hop_dst.GetIdx())
        all_dst_idx = list(set(two_and_one_hop_idx))
        if len(all_dst_idx) ==0: continue
        all_dst_idx.remove(src_idx)
        all_src_idx = [src_idx] *len(all_dst_idx)
        edges_src.extend(all_src_idx)
        edges_dst.extend(all_dst_idx)
    graph = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=lig.GetNumAtoms(), idtype=torch.long)
    graph.edata['feat'] = torch.from_numpy(np.linalg.norm(coords[edges_src] - coords[edges_dst], axis=1).astype(np.float32))
    return graph

def isRingAromatic(mol, bondRing):
    for id in bondRing:
        if not mol.GetBondWithIdx(id).GetIsAromatic():
            return False
    return True

def get_geometry_graph_ring(lig):
    coords = lig.GetConformer().GetPositions()
    rings = lig.GetRingInfo().AtomRings()
    bond_rings = lig.GetRingInfo().BondRings()
    edges_src = []
    edges_dst = []
    for i, atom in enumerate(lig.GetAtoms()):
        src_idx = atom.GetIdx()
        assert src_idx == i
        one_hop_dsts = [neighbor for neighbor in list(atom.GetNeighbors())]
        two_and_one_hop_idx = [neighbor.GetIdx() for neighbor in one_hop_dsts]
        for one_hop_dst in one_hop_dsts:
            for two_hop_dst in one_hop_dst.GetNeighbors():
                two_and_one_hop_idx.append(two_hop_dst.GetIdx())
        all_dst_idx = list(set(two_and_one_hop_idx))
        for ring_idx, ring in enumerate(rings):
            if src_idx in ring and isRingAromatic(lig,bond_rings[ring_idx]):
                all_dst_idx.extend(list(ring))
        all_dst_idx = list(set(all_dst_idx))
        if len(all_dst_idx) == 0: continue
        all_dst_idx.remove(src_idx)
        all_src_idx = [src_idx] *len(all_dst_idx)
        edges_src.extend(all_src_idx)
        edges_dst.extend(all_dst_idx)
    graph = dgl.graph((torch.tensor(edges_src), torch.tensor(edges_dst)), num_nodes=lig.GetNumAtoms(), idtype=torch.long)
    graph.edata['feat'] = torch.from_numpy(np.linalg.norm(coords[edges_src] - coords[edges_dst], axis=1).astype(np.float32))
    return graph

def get_lig_graph_multiple_conformer(mol, name, radius=20, max_neighbors=None, use_rdkit_coords=False, num_confs=10):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    try:
        count = 0
        success = False
        while not success:
            try:
                all_lig_coords = get_multiple_rdkit_coords_individual(mol,num_conf=num_confs)
                success = True
            except Exception as e:
                print(f'failed RDKit coordinate generation. Trying the {count}th time.')
                if count > 5:
                    raise Exception(e)
                count +=1

    except Exception as e:
        all_lig_coords = [true_lig_coords] * num_confs
        with open('temp_create_dataset_rdkit.log', 'a') as f:
            f.write('Generating RDKit conformer failed for  \n')
            f.write(name)
            f.write('\n')
            f.write(str(e))
            f.write('\n')
            f.flush()
        print('Generating RDKit conformer failed for  ')
        print(name)
        print(str(e))
    lig_graphs = []
    for i in range(num_confs):
        R, t = rigid_transform_Kabsch_3D(all_lig_coords[i].T, true_lig_coords.T)
        lig_coords = ((R @ (all_lig_coords[i]).T).T + t.squeeze())
        log('kabsch RMSD between rdkit ligand and true ligand is ',
            np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())

        num_nodes = lig_coords.shape[0]
        assert lig_coords.shape[1] == 3
        distance = spa.distance.cdist(lig_coords, lig_coords)

        src_list = []
        dst_list = []
        dist_list = []
        mean_norm_list = []
        for i in range(num_nodes):
            dst = list(np.where(distance[i, :] < radius)[0])
            dst.remove(i)
            if max_neighbors != None and len(dst) > max_neighbors:
                dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
            if len(dst) == 0:
                dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
                log(
                    f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
            assert i not in dst
            src = [i] * len(dst)
            src_list.extend(src)
            dst_list.extend(dst)
            valid_dist = list(distance[i, dst])
            dist_list.extend(valid_dist)
            valid_dist_np = distance[i, dst]
            sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
            weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
            assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
            diff_vecs = lig_coords[src, :] - lig_coords[dst, :]  # (neigh_num, 3)
            mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
            denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
            mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)

            mean_norm_list.append(mean_vec_ratio_norm)
        assert len(src_list) == len(dst_list)
        assert len(dist_list) == len(dst_list)
        graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

        graph.ndata['feat'] = lig_atom_featurizer(mol)
        graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
        graph.ndata['x'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
        graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
        if use_rdkit_coords:
            graph.ndata['new_x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
        lig_graphs.append(graph)
    return lig_graphs

def get_lig_graph_revised(mol, name, radius=20, max_neighbors=None, use_rdkit_coords=False):
    conf = mol.GetConformer()
    true_lig_coords = conf.GetPositions()
    if use_rdkit_coords:
        try:
            rdkit_coords = get_rdkit_coords(mol).numpy()
            R, t = rigid_transform_Kabsch_3D(rdkit_coords.T, true_lig_coords.T)
            lig_coords = ((R @ (rdkit_coords).T).T + t.squeeze())
            log('kabsch RMSD between rdkit ligand and true ligand is ', np.sqrt(np.sum((lig_coords - true_lig_coords) ** 2, axis=1).mean()).item())
        except Exception as e:
            lig_coords = true_lig_coords
            with open('temp_create_dataset_rdkit_timesplit_no_lig_or_rec_overlap_train.log', 'a') as f:
                f.write('Generating RDKit conformer failed for  \n')
                f.write(name)
                f.write('\n')
                f.write(str(e))
                f.write('\n')
                f.flush()
            print('Generating RDKit conformer failed for  ')
            print(name)
            print(str(e))
    else:
        lig_coords = true_lig_coords
    num_nodes = lig_coords.shape[0]
    assert lig_coords.shape[1] == 3
    distance = spa.distance.cdist(lig_coords, lig_coords)

    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_nodes):
        dst = list(np.where(distance[i, :] < radius)[0])
        dst.remove(i)
        if max_neighbors != None and len(dst) > max_neighbors:
            dst = list(np.argsort(distance[i, :]))[1: max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(distance[i, :]))[1:2]  # closest would be the index i itself > self loop
            log(
                f'The lig_radius {radius} was too small for one lig atom such that it had no neighbors. So we connected {i} to the closest other lig atom {dst}')
        assert i not in dst
        assert dst != []
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distance[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distance[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = lig_coords[src, :] - lig_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)

        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_nodes, idtype=torch.int32)

    graph.ndata['feat'] = lig_atom_featurizer(mol)
    graph.edata['feat'] = distance_featurizer(dist_list, 0.75)  # avg distance = 1.3 So divisor = (4/7)*1.3 = ~0.75
    graph.ndata['x'] = torch.from_numpy(np.array(true_lig_coords).astype(np.float32))
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    if use_rdkit_coords:
        graph.ndata['new_x'] = torch.from_numpy(np.array(lig_coords).astype(np.float32))
    return graph


def distance_featurizer(dist_list, divisor) -> torch.Tensor:
    # you want to use a divisor that is close to 4/7 times the average distance that you want to encode
    length_scale_list = [1.5 ** x for x in range(15)]
    center_list = [0. for _ in range(15)]

    num_edge = len(dist_list)
    dist_list = np.array(dist_list)

    transformed_dist = [np.exp(- ((dist_list / divisor) ** 2) / float(length_scale))
                        for length_scale, center in zip(length_scale_list, center_list)]

    transformed_dist = np.array(transformed_dist).T
    transformed_dist = transformed_dist.reshape((num_edge, -1))
    return torch.from_numpy(transformed_dist.astype(np.float32))


def get_hierarchical_graph(rec, rec_coords_list, c_alpha_coords, n_coords, c_coords, c_alpha_cutoff=20,
                           c_alpha_max_neighbors=None,
                           surface_graph_cutoff=10, surface_max_neighbors=None,
                           surface_mesh_cutoff=1.72):
    surface_mesh = get_surface(rec, 'msms -density 1')
    rec_coords_concat = np.concatenate(rec_coords_list, axis=0)
    distances = spatial.distance.cdist(rec_coords_concat, surface_mesh)
    # surface_indices = sorted(list(set(np.argmin(distances, axis=0)))) # use the closest atom instead
    surface_indices = sorted(list(set(np.where(distances < surface_mesh_cutoff)[0])))
    np_surface_indices = np.array(surface_indices)

    c_alpha_to_surface_src = []
    c_alpha_to_surface_dst = []

    c_alpha_to_surface_distances = []
    n_i_list = []
    u_i_list = []
    v_i_list = []
    atom_count = 0
    for i, res_coords in enumerate(rec_coords_list):
        res_indices = np.arange(len(res_coords)) + atom_count
        atom_count += len(res_coords)

        # get indices where the surface atom indices of this residue appear in surface_indices (CAREFUL: for this to work, the surface_indices have to be sorted)
        index_in_surface_atoms = np.where(np.isin(surface_indices, res_indices))[0]

        res_surface_indices = np_surface_indices[index_in_surface_atoms]
        c_alpha_to_surface_src.extend(len(index_in_surface_atoms) * [i])
        c_alpha_to_surface_dst.extend(list(index_in_surface_atoms))
        res_surface_coords = rec_coords_concat[res_surface_indices]
        nitrogen = n_coords[i]
        c_alpha = c_alpha_coords[i]
        carbon = c_coords[i]
        c_alpha_to_surface_distances.extend(list(np.linalg.norm((res_surface_coords - c_alpha), axis=1)))

        u_i = (nitrogen - c_alpha) / np.linalg.norm(nitrogen - c_alpha)
        t_i = (carbon - c_alpha) / np.linalg.norm(carbon - c_alpha)
        n_i = np.cross(u_i, t_i) / np.linalg.norm(np.cross(u_i, t_i))
        v_i = np.cross(n_i, u_i)
        assert (math.fabs(
            np.linalg.norm(v_i) - 1.) < 1e-5), "protein utils protein_to_graph_dips, v_i norm larger than 1"
        n_i_list.append(n_i)
        u_i_list.append(u_i)
        v_i_list.append(v_i)

    n_i_feat = np.stack(n_i_list, axis=0)
    u_i_feat = np.stack(u_i_list, axis=0)
    v_i_feat = np.stack(v_i_list, axis=0)
    num_residues = len(rec_coords_list)
    if num_residues <= 1:
        raise ValueError(f"l_or_r contains only 1 residue!")
    ################### Build the k-NN graph ##############################

    surface_coords = rec_coords_concat[surface_indices]
    surface_distances = spa.distance.cdist(surface_coords, surface_coords)
    surface_src = []
    surface_dst = []
    surface_edge_distances = []
    surface_mean_norms = []
    for i in range(len(surface_coords)):
        dst = list(np.where(surface_distances[i, :] < surface_graph_cutoff)[0])
        dst.remove(i)
        if surface_max_neighbors != None and len(dst) > surface_max_neighbors:
            dst = list(np.argsort(surface_distances[i, :]))[1: surface_max_neighbors + 1]  # closest would be self loop
        if len(dst) == 0:
            dst = list(np.argsort(surface_distances[i, :]))[1:2]  # closest would be the index i itself > self loop
            log(
                f'The surface_graph_cutoff {surface_graph_cutoff} was too small for one surface atom such that it had no neighbors. So we connected {i} to the closest other surface_atom {dst}')
        assert i not in dst
        src = [i] * len(dst)
        surface_src.extend(src)
        surface_dst.extend(dst)
        valid_dist = list(surface_distances[i, dst])
        surface_edge_distances.extend(valid_dist)

        valid_dist_np = surface_distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = surface_coords[src, :] - surface_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        surface_mean_norms.append(mean_vec_ratio_norm)
    assert len(surface_src) == len(surface_dst)
    assert len(surface_edge_distances) == len(surface_dst)

    c_alpha_distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    c_alpha_src = []
    c_alpha_dst = []
    c_alpha_edge_distances = []
    c_alpha_mean_norms = []
    for i in range(num_residues):
        dst = list(np.where(c_alpha_distances[i, :] < c_alpha_cutoff)[0])
        dst.remove(i)
        if c_alpha_max_neighbors != None and len(dst) > c_alpha_max_neighbors:
            dst = list(np.argsort(c_alpha_distances[i, :]))[1: c_alpha_max_neighbors + 1]
        if len(dst) == 0:
            dst = list(np.argsort(c_alpha_distances[i, :]))[1:2]  # choose second because first is i itself
            log(
                f'The c_alpha_cutoff {c_alpha_cutoff} was too small for one c_alpha such that it had no neighbors. So we connected it to the closest other c_alpha')
        assert i not in dst

        src = [i] * len(dst)
        c_alpha_src.extend(src)
        c_alpha_dst.extend(dst)
        valid_dist = list(c_alpha_distances[i, dst])
        c_alpha_edge_distances.extend(valid_dist)
        valid_dist_np = c_alpha_distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = c_alpha_coords[src, :] - c_alpha_coords[dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        c_alpha_mean_norms.append(mean_vec_ratio_norm)
    assert len(c_alpha_src) == len(c_alpha_dst)
    assert len(c_alpha_edge_distances) == len(c_alpha_dst)

    # Loop over all edges of the graph and build the various p_ij, q_ij, k_ij, t_ij pairs
    edge_feat_ori_list = []
    for i in range(len(c_alpha_edge_distances)):
        src = c_alpha_src[i]
        dst = c_alpha_dst[i]
        # place n_i, u_i, v_i as lines in a 3x3 basis matrix
        basis_matrix = np.stack((n_i_feat[dst, :], u_i_feat[dst, :], v_i_feat[dst, :]), axis=0)
        p_ij = np.matmul(basis_matrix, c_alpha_coords[src, :] - c_alpha_coords[dst, :])
        q_ij = np.matmul(basis_matrix, n_i_feat[src, :])  # shape (3,)
        k_ij = np.matmul(basis_matrix, u_i_feat[src, :])
        t_ij = np.matmul(basis_matrix, v_i_feat[src, :])
        s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)  # shape (12,)
        edge_feat_ori_list.append(s_ij)
    edge_feat_ori_feat = np.stack(edge_feat_ori_list, axis=0)  # shape (num_edges, 12)
    edge_feat_ori_feat = torch.from_numpy(edge_feat_ori_feat.astype(np.float32))

    c_alpha_edge_feat = torch.cat([distance_featurizer(c_alpha_edge_distances, divisor=3.5),
                                   edge_feat_ori_feat], axis=1)
    c_alpha_to_surface_feat = torch.cat([distance_featurizer(c_alpha_to_surface_distances, divisor=3.5),
                                         torch.zeros(len(c_alpha_to_surface_dst), 12)], axis=1)
    surface_edge_feat = torch.cat([distance_featurizer(surface_edge_distances, divisor=3.5),
                                   torch.zeros(len(surface_dst), 12)], axis=1)

    src = torch.cat([torch.tensor(c_alpha_src),
                     torch.tensor(c_alpha_to_surface_src),
                     torch.tensor(surface_src) + num_residues])
    dst = torch.cat([torch.tensor(c_alpha_dst),
                     torch.tensor(c_alpha_to_surface_dst) + num_residues,
                     torch.tensor(surface_dst) + num_residues])
    graph = dgl.graph((src, dst), num_nodes=num_residues + len(surface_indices), idtype=torch.int32)

    c_alpha_feat, surface_atom_feat = rec_atom_featurizer(rec, surface_indices)
    graph.ndata['feat'] = torch.cat([c_alpha_feat, surface_atom_feat], dim=0)
    graph.edata['feat'] = torch.cat([c_alpha_edge_feat, c_alpha_to_surface_feat, surface_edge_feat], dim=0)
    graph.ndata['x'] = torch.cat(
        [torch.tensor(c_alpha_coords, dtype=torch.float32), torch.tensor(surface_coords, dtype=torch.float32)], dim=0)
    graph.ndata['mu_r_norm'] = torch.tensor(np.concatenate([np.stack(c_alpha_mean_norms, axis=0),
                                                            np.stack(surface_mean_norms, axis=0)]), dtype=torch.float32)

    log(
        f'receptor num c_alphas: {len(c_alpha_feat)} - num surface atoms {len(surface_atom_feat)} - num c_alpha_edges {len(c_alpha_src)} - num cross_edges {len(c_alpha_to_surface_dst)} - num surface_edges {len(surface_dst)}')
    return graph


def get_calpha_graph(rec, c_alpha_coords, n_coords, c_coords, cutoff=20, max_neighbor=None):
    ################## Extract 3D coordinates and n_i,u_i,v_i vectors of representative residues ################
    residue_representatives_loc_list = []
    n_i_list = []
    u_i_list = []
    v_i_list = []
    for i, residue in enumerate(rec.get_residues()):
        n_coord = n_coords[i]
        c_alpha_coord = c_alpha_coords[i]
        c_coord = c_coords[i]
        u_i = (n_coord - c_alpha_coord) / np.linalg.norm(n_coord - c_alpha_coord)
        t_i = (c_coord - c_alpha_coord) / np.linalg.norm(c_coord - c_alpha_coord)
        n_i = np.cross(u_i, t_i) / np.linalg.norm(np.cross(u_i, t_i))
        v_i = np.cross(n_i, u_i)
        assert (math.fabs(
            np.linalg.norm(v_i) - 1.) < 1e-5), "protein utils protein_to_graph_dips, v_i norm larger than 1"
        n_i_list.append(n_i)
        u_i_list.append(u_i)
        v_i_list.append(v_i)
        residue_representatives_loc_list.append(c_alpha_coord)

    residue_representatives_loc_feat = np.stack(residue_representatives_loc_list, axis=0)  # (N_res, 3)
    n_i_feat = np.stack(n_i_list, axis=0)
    u_i_feat = np.stack(u_i_list, axis=0)
    v_i_feat = np.stack(v_i_list, axis=0)
    num_residues = len(c_alpha_coords)
    if num_residues <= 1:
        raise ValueError(f"rec contains only 1 residue!")

    ################### Build the k-NN graph ##############################
    assert num_residues == residue_representatives_loc_feat.shape[0]
    assert residue_representatives_loc_feat.shape[1] == 3
    distances = spa.distance.cdist(c_alpha_coords, c_alpha_coords)
    src_list = []
    dst_list = []
    dist_list = []
    mean_norm_list = []
    for i in range(num_residues):
        dst = list(np.where(distances[i, :] < cutoff)[0])
        dst.remove(i)
        if max_neighbor != None and len(dst) > max_neighbor:
            dst = list(np.argsort(distances[i, :]))[1: max_neighbor + 1]
        if len(dst) == 0:
            dst = list(np.argsort(distances[i, :]))[1:2]  # choose second because first is i itself
            log(
                f'The c_alpha_cutoff {cutoff} was too small for one c_alpha such that it had no neighbors. So we connected it to the closest other c_alpha')
        assert i not in dst
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
        valid_dist = list(distances[i, dst])
        dist_list.extend(valid_dist)
        valid_dist_np = distances[i, dst]
        sigma = np.array([1., 2., 5., 10., 30.]).reshape((-1, 1))
        weights = softmax(- valid_dist_np.reshape((1, -1)) ** 2 / sigma, axis=1)  # (sigma_num, neigh_num)
        assert weights[0].sum() > 1 - 1e-2 and weights[0].sum() < 1.01
        diff_vecs = residue_representatives_loc_feat[src, :] - residue_representatives_loc_feat[
                                                               dst, :]  # (neigh_num, 3)
        mean_vec = weights.dot(diff_vecs)  # (sigma_num, 3)
        denominator = weights.dot(np.linalg.norm(diff_vecs, axis=1))  # (sigma_num,)
        mean_vec_ratio_norm = np.linalg.norm(mean_vec, axis=1) / denominator  # (sigma_num,)
        mean_norm_list.append(mean_vec_ratio_norm)
    assert len(src_list) == len(dst_list)
    assert len(dist_list) == len(dst_list)
    graph = dgl.graph((torch.tensor(src_list), torch.tensor(dst_list)), num_nodes=num_residues, idtype=torch.int32)

    graph.ndata['feat'] = rec_residue_featurizer(rec)
    graph.edata['feat'] = distance_featurizer(dist_list, divisor=4)  # avg distance = 7. So divisor = (4/7)*7 = 4

    # Loop over all edges of the graph and build the various p_ij, q_ij, k_ij, t_ij pairs
    edge_feat_ori_list = []
    for i in range(len(dist_list)):
        src = src_list[i]
        dst = dst_list[i]
        # place n_i, u_i, v_i as lines in a 3x3 basis matrix
        basis_matrix = np.stack((n_i_feat[dst, :], u_i_feat[dst, :], v_i_feat[dst, :]), axis=0)
        p_ij = np.matmul(basis_matrix,
                         residue_representatives_loc_feat[src, :] - residue_representatives_loc_feat[
                                                                    dst, :])
        q_ij = np.matmul(basis_matrix, n_i_feat[src, :])  # shape (3,)
        k_ij = np.matmul(basis_matrix, u_i_feat[src, :])
        t_ij = np.matmul(basis_matrix, v_i_feat[src, :])
        s_ij = np.concatenate((p_ij, q_ij, k_ij, t_ij), axis=0)  # shape (12,)
        edge_feat_ori_list.append(s_ij)
    edge_feat_ori_feat = np.stack(edge_feat_ori_list, axis=0)  # shape (num_edges, 4, 3)
    edge_feat_ori_feat = torch.from_numpy(edge_feat_ori_feat.astype(np.float32))
    graph.edata['feat'] = torch.cat([graph.edata['feat'], edge_feat_ori_feat], axis=1)  # (num_edges, 17)

    residue_representatives_loc_feat = torch.from_numpy(residue_representatives_loc_feat.astype(np.float32))
    graph.ndata['x'] = residue_representatives_loc_feat
    graph.ndata['mu_r_norm'] = torch.from_numpy(np.array(mean_norm_list).astype(np.float32))
    return graph


def lig_rec_graphs_to_complex_graph(ligand_graph, receptor_graph):
    ll = [('lig', 'll', 'lig'), (ligand_graph.edges()[0], ligand_graph.edges()[1])]
    rr = [('rec', 'rr', 'rec'), (receptor_graph.edges()[0], receptor_graph.edges()[1])]
    # rl = [('rec', 'cross', 'lig'),(torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    # lr = [('lig', 'cross', 'rec'),(torch.tensor([], dtype=torch.int32), torch.tensor([], dtype=torch.int32))]
    num_nodes = {'lig': ligand_graph.num_nodes(), 'rec': receptor_graph.num_nodes()}
    # hetero_graph = dgl.heterograph({ll[0]: ll[1], rr[0]: rr[1], rl[0]: rl[1], lr[0]: lr[1]}, num_nodes_dict=num_nodes)
    hetero_graph = dgl.heterograph({ll[0]: ll[1], rr[0]: rr[1]}, num_nodes_dict=num_nodes)
    hetero_graph.nodes['lig'].data['feat'] = ligand_graph.ndata['feat']
    hetero_graph.nodes['lig'].data['x'] = ligand_graph.ndata['x']
    hetero_graph.nodes['lig'].data['new_x'] = ligand_graph.ndata['x']
    hetero_graph.nodes['lig'].data['mu_r_norm'] = ligand_graph.ndata['mu_r_norm']
    hetero_graph.edges['ll'].data['feat'] = ligand_graph.edata['feat']
    hetero_graph.nodes['rec'].data['feat'] = receptor_graph.ndata['feat']
    hetero_graph.nodes['rec'].data['x'] = receptor_graph.ndata['x']
    hetero_graph.nodes['rec'].data['mu_r_norm'] = receptor_graph.ndata['mu_r_norm']
    hetero_graph.edges['rr'].data['feat'] = receptor_graph.edata['feat']
    # # Add cross edges  (Ends up using a lot of GPU memory and is slow):
    # ligand_ids = torch.arange(num_nodes['lig'], dtype=torch.int32)
    # receptor_ids = torch.arange(num_nodes['rec'], dtype=torch.int32)
    # cross_src_lr = torch.tile(ligand_ids, (num_nodes['rec'],))
    # cross_src_rl = torch.tile(receptor_ids, (num_nodes['lig'],))
    # cross_dst_lr = torch.repeat_interleave(receptor_ids, num_nodes['lig'])
    # cross_dst_rl = torch.repeat_interleave(ligand_ids, num_nodes['rec'])
    # hetero_graph.add_edges(cross_src_lr, cross_dst_lr, etype=('lig', 'cross', 'rec'))
    # hetero_graph.add_edges(cross_src_rl, cross_dst_rl, etype=('rec', 'cross', 'lig'))
    return hetero_graph

def read_molecule(molecule_file, sanitize=False, calc_charges=False, remove_hs=False):
    """Load a molecule from a file of format ``.mol2`` or ``.sdf`` or ``.pdbqt`` or ``.pdb``.

    Parameters
    ----------
    molecule_file : str
        Path to file for storing a molecule, which can be of format ``.mol2`` or ``.sdf``
        or ``.pdbqt`` or ``.pdb``.
    sanitize : bool
        Whether sanitization is performed in initializing RDKit molecule instances. See
        https://www.rdkit.org/docs/RDKit_Book.html for details of the sanitization.
        Default to False.
    calc_charges : bool
        Whether to add Gasteiger charges via RDKit. Setting this to be True will enforce
        ``sanitize`` to be True. Default to False.
    remove_hs : bool
        Whether to remove hydrogens via RDKit. Note that removing hydrogens can be quite
        slow for large molecules. Default to False.
    use_conformation : bool
        Whether we need to extract molecular conformation from proteins and ligands.
        Default to True.

    Returns
    -------
    mol : rdkit.Chem.rdchem.Mol
        RDKit molecule instance for the loaded molecule.
    coordinates : np.ndarray of shape (N, 3) or None
        The 3D coordinates of atoms in the molecule. N for the number of atoms in
        the molecule. None will be returned if ``use_conformation`` is False or
        we failed to get conformation information.
    """
    if molecule_file.endswith('.mol2'):
        mol = Chem.MolFromMol2File(molecule_file, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.sdf'):
        supplier = Chem.SDMolSupplier(molecule_file, sanitize=False, removeHs=False)
        mol = supplier[0]
    elif molecule_file.endswith('.pdbqt'):
        with open(molecule_file) as file:
            pdbqt_data = file.readlines()
        pdb_block = ''
        for line in pdbqt_data:
            pdb_block += '{}\n'.format(line[:66])
        mol = Chem.MolFromPDBBlock(pdb_block, sanitize=False, removeHs=False)
    elif molecule_file.endswith('.pdb'):
        mol = Chem.MolFromPDBFile(molecule_file, sanitize=False, removeHs=False)
    else:
        return ValueError('Expect the format of the molecule_file to be '
                          'one of .mol2, .sdf, .pdbqt and .pdb, got {}'.format(molecule_file))

    try:
        if sanitize or calc_charges:
            Chem.SanitizeMol(mol)

        if calc_charges:
            # Compute Gasteiger charges on the molecule.
            try:
                AllChem.ComputeGasteigerCharges(mol)
            except:
                warnings.warn('Unable to compute charges for the molecule.')

        if remove_hs:
            mol = Chem.RemoveHs(mol, sanitize=sanitize)
    except:
        return None

    return mol
