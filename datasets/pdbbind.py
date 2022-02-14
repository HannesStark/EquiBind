import os
import random
from copy import deepcopy, copy

from dgl import save_graphs, load_graphs

from joblib import Parallel, delayed, cpu_count
import torch
import dgl
from biopandas.pdb import PandasPdb
from joblib.externals.loky import get_reusable_executor

from rdkit import Chem
from rdkit.Chem import MolFromPDBFile
from rdkit.Chem.rdmolops import GetAdjacencyMatrix
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch.nn.functional as F

from commons.geometry_utils import random_rotation_translation, rigid_transform_Kabsch_3D_torch
from commons.process_mols import get_rdkit_coords, get_receptor, get_pocket_coords, \
    read_molecule, get_rec_graph, get_lig_graph_revised, get_receptor_atom_subgraph, get_lig_structure_graph, \
    get_geometry_graph, get_lig_graph_multiple_conformer, get_geometry_graph_ring
from commons.utils import pmap_multi, read_strings_from_txt, log


class PDBBind(Dataset):
    """"""

    def __init__(self, device='cuda:0',
                 complex_names_path='data/',
                 bsp_proteins=False,
                 bsp_ligands=False,
                 pocket_cutoff=8.0,
                 use_rec_atoms=False,
                 n_jobs=None,
                 chain_radius=7,
                 c_alpha_max_neighbors=10,
                 lig_max_neighbors=20,
                 translation_distance=5.0,
                 lig_graph_radius=30,
                 rec_graph_radius=30,
                 surface_max_neighbors=5,
                 surface_graph_cutoff=5,
                 surface_mesh_cutoff=1.7,
                 deep_bsp_preprocessing=True,
                 only_polar_hydrogens=False,
                 use_rdkit_coords=False,
                 pocket_mode='match_terminal_atoms',
                 dataset_size=None,
                 remove_h=False,
                 rec_subgraph=False,
                 is_train_data=False,
                 min_shell_thickness=2,
                 subgraph_radius=10,
                 subgraph_max_neigbor=8,
                 subgraph_cutoff=4,
                 lig_structure_graph= False,
                 random_rec_atom_subgraph= False,
                 subgraph_augmentation=False,
                 lig_predictions_name=None,
                 geometry_regularization= False,
                 multiple_rdkit_conformers = False,
                 random_rec_atom_subgraph_radius= 10,
                 geometry_regularization_ring= False,
                 num_confs=10,
                 transform=None, **kwargs):
        # subset name is either 'pdbbind_filtered' or 'casf_test'
        self.chain_radius = chain_radius
        self.pdbbind_dir = 'data/PDBBind'
        self.bsp_dir = 'data/deepBSP'
        self.only_polar_hydrogens = only_polar_hydrogens
        self.complex_names_path = complex_names_path
        self.pocket_cutoff = pocket_cutoff
        self.use_rec_atoms = use_rec_atoms
        self.deep_bsp_preprocessing = deep_bsp_preprocessing
        self.device = device
        self.lig_graph_radius = lig_graph_radius
        self.rec_graph_radius = rec_graph_radius
        self.surface_max_neighbors = surface_max_neighbors
        self.surface_graph_cutoff = surface_graph_cutoff
        self.surface_mesh_cutoff = surface_mesh_cutoff
        self.dataset_size = dataset_size
        self.c_alpha_max_neighbors = c_alpha_max_neighbors
        self.lig_max_neighbors = lig_max_neighbors
        self.n_jobs = cpu_count() - 1 if n_jobs == None else n_jobs
        self.translation_distance = translation_distance
        self.pocket_mode = pocket_mode
        self.use_rdkit_coords = use_rdkit_coords
        self.bsp_proteins = bsp_proteins
        self.bsp_ligands = bsp_ligands
        self.remove_h = remove_h
        self.is_train_data = is_train_data
        self.subgraph_augmentation = subgraph_augmentation
        self.min_shell_thickness = min_shell_thickness
        self.rec_subgraph = rec_subgraph
        self.subgraph_radius = subgraph_radius
        self.subgraph_max_neigbor=subgraph_max_neigbor
        self.subgraph_cutoff=subgraph_cutoff
        self.random_rec_atom_subgraph = random_rec_atom_subgraph
        self.lig_structure_graph =lig_structure_graph
        self.random_rec_atom_subgraph_radius = random_rec_atom_subgraph_radius
        self.lig_predictions_name = lig_predictions_name
        self.geometry_regularization = geometry_regularization
        self.geometry_regularization_ring = geometry_regularization_ring
        self.multiple_rdkit_conformers = multiple_rdkit_conformers
        self.num_confs = num_confs
        self.conformer_id = 0
        if self.lig_predictions_name ==None:
            self.rec_subgraph_path = f'rec_subgraphs_cutoff{self.subgraph_cutoff}_radius{self.subgraph_radius}_maxNeigh{self.subgraph_max_neigbor}.pt'
        else:
            self.rec_subgraph_path = f'rec_subgraphs_cutoff{self.subgraph_cutoff}_radius{self.subgraph_radius}_maxNeigh{self.subgraph_max_neigbor}_{self.lig_predictions_name}'

        self.processed_dir = f'data/processed/size{self.dataset_size}_INDEX{os.path.splitext(os.path.basename(self.complex_names_path))[0]}_Hpolar{int(self.only_polar_hydrogens)}_H{int(not self.remove_h)}_BSPprot{int(self.bsp_proteins)}_BSPlig{int(self.bsp_ligands)}_surface{int(self.use_rec_atoms)}_pocketRad{self.pocket_cutoff}_ligRad{self.lig_graph_radius}_recRad{self.rec_graph_radius}_recMax{self.c_alpha_max_neighbors}_ligMax{self.lig_max_neighbors}_chain{self.chain_radius}_POCKET{self.pocket_mode}'
        print(f'using processed directory: {self.processed_dir}')
        if self.use_rdkit_coords:
            self.lig_graph_path = 'lig_graphs_rdkit_coords.pt'
        else:
            self.lig_graph_path = 'lig_graphs.pt'
        if self.multiple_rdkit_conformers:
            self.lig_graph_path = 'lig_graphs_rdkit_multiple_conformers.pt'
        if not os.path.exists('data/processed/'):
            os.mkdir('data/processed/')
        if (not os.path.exists(os.path.join(self.processed_dir, 'geometry_regularization.pt')) and self.geometry_regularization) or (not os.path.exists(os.path.join(self.processed_dir, 'geometry_regularization_ring.pt')) and self.geometry_regularization_ring) or not os.path.exists(os.path.join(self.processed_dir, 'rec_graphs.pt')) or not os.path.exists(os.path.join(self.processed_dir, 'pocket_and_rec_coords.pt')) or not os.path.exists(os.path.join(self.processed_dir, self.lig_graph_path)) or (not os.path.exists(os.path.join(self.processed_dir, self.rec_subgraph_path)) and self.rec_subgraph) or (not os.path.exists(os.path.join(self.processed_dir, 'lig_structure_graphs.pt')) and self.lig_structure_graph):
            self.process()
        log('loading data into memory')
        coords_dict = torch.load(os.path.join(self.processed_dir, 'pocket_and_rec_coords.pt'))
        self.pockets_coords = coords_dict['pockets_coords']
        self.lig_graphs, _ = load_graphs(os.path.join(self.processed_dir, self.lig_graph_path))
        if self.multiple_rdkit_conformers:
            self.lig_graphs = [self.lig_graphs[i:i + self.num_confs] for i in range(0, len(self.lig_graphs), self.num_confs)]
        self.rec_graphs, _ = load_graphs(os.path.join(self.processed_dir, 'rec_graphs.pt'))
        if self.rec_subgraph:
            self.rec_atom_subgraphs, _ = load_graphs(os.path.join(self.processed_dir, self.rec_subgraph_path))
        if self.lig_structure_graph:
            self.lig_structure_graphs, _ =  load_graphs(os.path.join(self.processed_dir, 'lig_structure_graphs.pt'))
            masks_angles = torch.load(os.path.join(self.processed_dir, 'torsion_masks_and_angles.pt'))
            self.angles = masks_angles['angles']
            self.masks = masks_angles['masks']
        if self.geometry_regularization:
            print(os.path.join(self.processed_dir, 'geometry_regularization.pt'))
            self.geometry_graphs, _ =  load_graphs(os.path.join(self.processed_dir, 'geometry_regularization.pt'))
        if self.geometry_regularization_ring:
            print(os.path.join(self.processed_dir, 'geometry_regularization_ring.pt'))
            self.geometry_graphs, _ =  load_graphs(os.path.join(self.processed_dir, 'geometry_regularization_ring.pt'))
        self.complex_names = coords_dict['complex_names']
        assert len(self.lig_graphs) == len(self.rec_graphs)
        log('finish loading data into memory')
        self.cache = {}


    def __len__(self):
        return len(self.lig_graphs)

    def __getitem__(self, idx):
        pocket_coords = self.pockets_coords[idx]
        if self.lig_structure_graph:
            lig_graph = deepcopy(self.lig_structure_graphs[idx])
        else:
            if self.multiple_rdkit_conformers:
                lig_graph = deepcopy(self.lig_graphs[idx][self.conformer_id])
            else:
                lig_graph = deepcopy(self.lig_graphs[idx])
        lig_coords = lig_graph.ndata['x']
        rec_graph = self.rec_graphs[idx]

        # Randomly rotate and translate the ligand.
        rot_T, rot_b = random_rotation_translation(translation_distance=self.translation_distance)
        if self.use_rdkit_coords:
            lig_coords_to_move =lig_graph.ndata['new_x']
        else:
            lig_coords_to_move = lig_coords
        mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
        lig_graph.ndata['new_x'] = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
        new_pocket_coords = (rot_T @ (pocket_coords - mean_to_remove).T).T + rot_b

        if self.subgraph_augmentation and self.is_train_data:
            with torch.no_grad():
                if idx in self.cache:
                    max_distance, min_distance, distances = self.cache[idx]
                else:
                    lig_centroid = lig_graph.ndata['x'].mean(dim=0)
                    distances = torch.norm(rec_graph.ndata['x'] - lig_centroid, dim=1)
                    max_distance = torch.max(distances)
                    min_distance = torch.min(distances)
                    self.cache[idx] = (min_distance.item(), max_distance.item(), distances)
                radius = min_distance + self.min_shell_thickness + random.random() * abs((
                            max_distance - min_distance - self.min_shell_thickness))
                rec_graph = dgl.node_subgraph(rec_graph, distances <= radius)
                assert rec_graph.num_nodes() > 0
        if self.rec_subgraph:
            rec_graph = self.rec_atom_subgraphs[idx]
            if self.random_rec_atom_subgraph:
                rot_T, rot_b = random_rotation_translation(translation_distance=2)
                translated_lig_coords = lig_coords + rot_b
                min_distances, _ = torch.cdist(rec_graph.ndata['x'],translated_lig_coords).min(dim=1)
                rec_graph = dgl.node_subgraph(rec_graph, min_distances < self.random_rec_atom_subgraph_radius)
                assert rec_graph.num_nodes() > 0

        geometry_graph = self.geometry_graphs[idx] if self.geometry_regularization or self.geometry_regularization_ring else None
        if self.lig_structure_graph:
            return lig_graph.to(self.device), rec_graph.to(self.device), self.masks[idx], self.angles[idx], lig_coords, rec_graph.ndata['x'], new_pocket_coords, pocket_coords,geometry_graph, self.complex_names[idx], idx
        else:
            return lig_graph.to(self.device), rec_graph.to(self.device), lig_coords, rec_graph.ndata['x'], new_pocket_coords, pocket_coords, geometry_graph, self.complex_names[idx], idx

    def process(self):
        log(f'Processing complexes from [{self.complex_names_path}] and saving it to [{self.processed_dir}]')

        complex_names = read_strings_from_txt(self.complex_names_path)
        if self.dataset_size != None:
            complex_names = complex_names[:self.dataset_size]
        if (self.remove_h or self.only_polar_hydrogens) and '4acu' in complex_names:
            complex_names.remove('4acu')  # in this complex's ligand the hydrogens cannot be removed
        log(f'Loading {len(complex_names)} complexes.')
        ligs = []
        to_remove = []
        for name in tqdm(complex_names, desc='loading ligands'):
            if self.bsp_ligands:
                lig = read_molecule(os.path.join(self.bsp_dir, name, f'Lig_native.pdb'), sanitize=True, remove_hs=self.remove_h)
                if lig == None:
                    to_remove.append(name)
                    continue
            else:
                lig = read_molecule(os.path.join(self.pdbbind_dir, name, f'{name}_ligand.sdf'), sanitize=True,
                                    remove_hs=self.remove_h)
                if lig == None:  # read mol2 file if sdf file cannot be sanitized
                    lig = read_molecule(os.path.join(self.pdbbind_dir, name, f'{name}_ligand.mol2'), sanitize=True,
                                        remove_hs=self.remove_h)
            if self.only_polar_hydrogens:
                for atom in lig.GetAtoms():
                    if atom.GetAtomicNum() == 1 and [x.GetAtomicNum() for x in atom.GetNeighbors()] == [6]:
                        atom.SetAtomicNum(0)
                lig = Chem.DeleteSubstructs(lig, Chem.MolFromSmarts('[#0]'))
                Chem.SanitizeMol(lig)
            ligs.append(lig)
        for name in to_remove:
            complex_names.remove(name)

        if self.bsp_proteins:
            rec_paths = [os.path.join(self.bsp_dir, name, f'Rec.pdb') for name in complex_names]
        else:
            rec_paths = [os.path.join(self.pdbbind_dir, name, f'{name}_protein_processed.pdb') for name in
                         complex_names]

        if not os.path.exists(self.processed_dir):
            os.mkdir(self.processed_dir)

        if not os.path.exists(os.path.join(self.processed_dir, 'rec_graphs.pt')) or not os.path.exists(os.path.join(self.processed_dir, 'pocket_and_rec_coords.pt')) or (not os.path.exists(os.path.join(self.processed_dir, self.rec_subgraph_path)) and self.rec_subgraph):
            log('Get receptors, filter chains, and get its coordinates')
            receptor_representatives = pmap_multi(get_receptor, zip(rec_paths, ligs), n_jobs=self.n_jobs, cutoff=self.chain_radius, desc='Get receptors')
            recs, recs_coords, c_alpha_coords, n_coords, c_coords = map(list, zip(*receptor_representatives))
            # rec coords is a list with n_residues many arrays of shape: [n_atoms_in_residue, 3]


        if not os.path.exists(os.path.join(self.processed_dir, 'pocket_and_rec_coords.pt')):
            log('Get Pocket Coordinates')
            pockets_coords = pmap_multi(get_pocket_coords, zip(ligs, recs_coords), n_jobs=self.n_jobs,
                                        cutoff=self.pocket_cutoff, pocket_mode=self.pocket_mode,
                                        desc='Get pocket coords')
            recs_coords_concat = [torch.tensor(np.concatenate(rec_coords, axis=0)) for rec_coords in recs_coords]
            torch.save({'pockets_coords': pockets_coords,
                        'all_rec_coords': recs_coords_concat,
                        # coords of all atoms and not only those included in graph
                        'complex_names': complex_names,
                        }, os.path.join(self.processed_dir, 'pocket_and_rec_coords.pt'))
        else:
            log('pocket_and_rec_coords.pt already exists. Using those instead of creating new ones.')

        if not os.path.exists(os.path.join(self.processed_dir, 'rec_graphs.pt')):
            log('Get receptor Graphs')
            rec_graphs = pmap_multi(get_rec_graph,
                                    zip(recs, recs_coords, c_alpha_coords, n_coords, c_coords), n_jobs=self.n_jobs,
                                    use_rec_atoms=self.use_rec_atoms, rec_radius=self.rec_graph_radius,
                                    surface_max_neighbors=self.surface_max_neighbors,
                                    surface_graph_cutoff=self.surface_graph_cutoff,
                                    surface_mesh_cutoff=self.surface_mesh_cutoff,
                                    c_alpha_max_neighbors=self.c_alpha_max_neighbors,
                                    desc='Convert receptors to graphs')
            save_graphs(os.path.join(self.processed_dir, 'rec_graphs.pt'), rec_graphs)
        else:
            log('rec_graphs.pt already exists. Using those instead of creating new ones.')
        log('Done converting to graphs')

        if self.lig_predictions_name != None:
            ligs_coords = torch.load(os.path.join('data/processed', self.lig_predictions_name))['predictions'][:len(ligs)]
        else:
            ligs_coords = [None] * len(ligs)
        if self.rec_subgraph and not os.path.exists(os.path.join(self.processed_dir, self.rec_subgraph_path)):
            log('Get receptor subgraphs')
            rec_subgraphs = pmap_multi(get_receptor_atom_subgraph,
                                    zip(recs, recs_coords, ligs, ligs_coords), n_jobs=self.n_jobs,
                                    max_neighbor=self.subgraph_max_neigbor, subgraph_radius=self.subgraph_radius,
                                    graph_cutoff=self.subgraph_cutoff,
                                    desc='get receptor subgraphs')
            save_graphs(os.path.join(self.processed_dir, self.rec_subgraph_path), rec_subgraphs)
        else:
            log(os.path.join(self.processed_dir, self.rec_subgraph_path), ' already exists. Using those instead of creating new ones.')
        log('Done creating receptor subgraphs')

        if not os.path.exists(os.path.join(self.processed_dir, self.lig_graph_path)):
            log('Convert ligands to graphs')
            if self.multiple_rdkit_conformers:
                lig_graphs = pmap_multi(get_lig_graph_multiple_conformer, zip(ligs,complex_names), n_jobs=self.n_jobs,
                                    max_neighbors=self.lig_max_neighbors, use_rdkit_coords=self.use_rdkit_coords,
                                    radius=self.lig_graph_radius, num_confs=self.num_confs, desc='Convert ligands to graphs')
                lig_graphs = [item for sublist in lig_graphs for item in sublist]
            else:
                lig_graphs = pmap_multi(get_lig_graph_revised, zip(ligs,complex_names), n_jobs=self.n_jobs,
                                    max_neighbors=self.lig_max_neighbors, use_rdkit_coords=self.use_rdkit_coords,
                                    radius=self.lig_graph_radius, desc='Convert ligands to graphs')

            save_graphs(os.path.join(self.processed_dir, self.lig_graph_path), lig_graphs)
        else:
            log('lig_graphs.pt already exists. Using those instead of creating new ones.')

        if not os.path.exists(os.path.join(self.processed_dir, 'lig_structure_graphs.pt')) and self.lig_structure_graph:
            log('Convert ligands to graphs')
            graphs_masks_angles = pmap_multi(get_lig_structure_graph, zip(ligs), n_jobs=self.n_jobs, desc='Get ligand structure graphs with angle information')
            graphs, masks, angles = map(list, zip(*graphs_masks_angles))
            torch.save({'masks': masks,
                        'angles': angles,
                        }, os.path.join(self.processed_dir, 'torsion_masks_and_angles.pt'))
            save_graphs(os.path.join(self.processed_dir, 'lig_structure_graphs.pt'), graphs)
        else:
            log('lig_structure_graphs.pt already exists or is not needed.')

        if not os.path.exists(os.path.join(self.processed_dir, 'geometry_regularization.pt')):
            log('Convert ligands to geometry graph')
            geometry_graphs = [get_geometry_graph(lig) for lig in ligs]
            save_graphs(os.path.join(self.processed_dir, 'geometry_regularization.pt'), geometry_graphs)
        else:
            log('geometry_regularization.pt already exists or is not needed.')

        if not os.path.exists(os.path.join(self.processed_dir, 'geometry_regularization_ring.pt')):
            log('Convert ligands to geometry graph')
            geometry_graphs = [get_geometry_graph_ring(lig) for lig in ligs]
            save_graphs(os.path.join(self.processed_dir, 'geometry_regularization_ring.pt'), geometry_graphs)
        else:
            log('geometry_regularization.pt already exists or is not needed.')

        get_reusable_executor().shutdown(wait=True)
