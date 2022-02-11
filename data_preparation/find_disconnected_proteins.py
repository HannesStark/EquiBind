
import glob
import os

import networkx as nx
from biopandas.pdb import PandasPdb

from scipy import spatial
from tqdm import tqdm
import numpy as np
import pandas as pd

from commons.utils import write_strings_to_txt


pdb_path = 'data/PDBBind'
casf_names = os.listdir('data/deepBSP/casf_test')
bsp_names = os.listdir('data/deepBSP/pdbbind_filtered')
pdbbind_names = os.listdir(pdb_path)

df_pdb_id = pd.read_csv('data/PDBbind_index/INDEX_general_PL_name.2020', sep="  ", comment='#', header=None, names=['complex_name', 'year', 'pdb_id', 'd', 'e','f','g','h','i','j','k','l','m','n','o'])
df_pdb_id = df_pdb_id[['complex_name','year','pdb_id']]

df_data = pd.read_csv('data/PDBbind_index/INDEX_general_PL_data.2020', sep="  ", comment='#', header=None, names=['complex_name','resolution','year', 'logkd', 'kd', 'reference', 'ligand_name', 'a', 'b', 'c'])
df_data = df_data[['complex_name','resolution','year', 'logkd', 'kd', 'reference', 'ligand_name']]

cutoff = 5
connected = []
for name in tqdm(pdbbind_names):
    df = PandasPdb().read_pdb(os.path.join(pdb_path, name, f'{name}_protein_obabel_reduce.pdb')).df['ATOM']
    df.rename(columns={'chain_id': 'chain', 'residue_number': 'residue', 'residue_name': 'resname',
                       'x_coord': 'x', 'y_coord': 'y', 'z_coord': 'z', 'element_symbol': 'element'}, inplace=True)
    df = list(df.groupby(['chain']))  ## Not the same as sequence order !

    chain_coords_list = []
    for chain in df:
        chain_coords_list.append(chain[1][['x', 'y', 'z']].to_numpy().squeeze().astype(np.float32))

    num_chains = len(chain_coords_list)
    distance = np.full((num_chains, num_chains), -np.inf)
    for i in range(num_chains - 1):
        for j in range((i + 1), num_chains):
            pairwise_dis = spatial.distance.cdist(chain_coords_list[i],chain_coords_list[j])
            distance[i, j] = np.min(pairwise_dis)
            distance[j, i] = np.min(pairwise_dis)
    src_list = []
    dst_list = []
    for i in range(num_chains):
        dst = list(np.where(distance[i, :] < cutoff)[0])
        src = [i] * len(dst)
        src_list.extend(src)
        dst_list.extend(dst)
    graph = nx.Graph()
    graph.add_edges_from(zip(src_list, dst_list))
    if nx.is_connected(graph):
        connected.append(name)
    else:
        print(f'not connected: {name}')
write_strings_to_txt(connected, f'data/complex_names_connected_by_{cutoff}')
print(len(connected))