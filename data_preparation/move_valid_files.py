import os
from shutil import copyfile

from tqdm import tqdm

from commons.utils import read_strings_from_txt

data_path = '../data/PDBBind'
overwrite = False
names = sorted(os.listdir(data_path))
invalid_names = read_strings_from_txt('select_chains.log')

valid_names = list(set(names) - set(invalid_names))

if not os.path.exists('../data/PDBBind_processed'):
    os.mkdir('../data/PDBBind_processed')
for i, name in tqdm(enumerate(valid_names)):
    if not os.path.exists(f'../data/PDBBind_processed/{name}'):
        os.mkdir(f'../data/PDBBind_processed/{name}')
    rec_path = os.path.join(data_path, name, f'{name}_protein.pdb')

    copyfile(os.path.join(data_path, name, f'{name}_protein_processed.pdb'), f'../data/PDBBind_processed/{name}/{name}_protein_processed.pdb')
    copyfile(os.path.join(data_path, name, f'{name}_ligand.mol2'), f'../data/PDBBind_processed/{name}/{name}_ligand.mol2')
    copyfile(os.path.join(data_path, name, f'{name}_ligand.sdf'),
             f'../data/PDBBind_processed/{name}/{name}_ligand.sdf')


