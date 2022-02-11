# you need openbabel installed to use this (can be installed with anaconda)
import os
import subprocess

import time

from tqdm import tqdm

start_time = time.time()
data_path = 'data/PDBBind'
overwrite = False
names = sorted(os.listdir(data_path))

for i, name in tqdm(enumerate(names)):
    rec_path = os.path.join(data_path, name, f'{name}_protein.pdb')
    return_code = subprocess.run(
        f"obabel {rec_path} -O{os.path.join(data_path, name, f'{name}_protein_obabel.pdb')}", shell=True)
    print(return_code)


print("--- %s seconds ---" % (time.time() - start_time))
