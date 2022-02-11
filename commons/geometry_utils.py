import copy
import math

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolTransforms
from scipy.spatial.transform import Rotation


def random_rotation_translation(translation_distance):
    rotation = Rotation.random(num=1)
    rotation_matrix = rotation.as_matrix().squeeze()

    t = np.random.randn(1, 3)
    t = t / np.sqrt( np.sum(t * t))
    length = np.random.uniform(low=0, high=translation_distance)
    t = t * length
    return torch.from_numpy(rotation_matrix.astype(np.float32)), torch.from_numpy(t.astype(np.float32))

# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_Kabsch_3D(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = np.mean(A, axis=1, keepdims=True)
    centroid_B = np.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = np.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = np.diag([1.,1.,-1.])
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(np.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t

# R = 3x3 rotation matrix
# t = 3x1 column vector
# This already takes residue identity into account.
def rigid_transform_Kabsch_3D_torch(A, B):
    assert A.shape[1] == B.shape[1]
    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")
    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")


    # find mean column wise: 3 x 1
    centroid_A = torch.mean(A, axis=1, keepdims=True)
    centroid_B = torch.mean(B, axis=1, keepdims=True)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ Bm.T

    # find rotation
    U, S, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T

    # special reflection case
    if torch.linalg.det(R) < 0:
        # print("det(R) < R, reflection detected!, correcting for it ...")
        SS = torch.diag(torch.tensor([1.,1.,-1.], device=A.device))
        R = (Vt.T @ SS) @ U.T
    assert math.fabs(torch.linalg.det(R) - 1) < 1e-5

    t = -R @ centroid_A + centroid_B
    return R, t


def get_torsions(mol_list):
    atom_counter = 0
    torsionList = []
    dihedralList = []
    for m in mol_list:
        torsionSmarts = '[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]'
        torsionQuery = Chem.MolFromSmarts(torsionSmarts)
        matches = m.GetSubstructMatches(torsionQuery)
        conf = m.GetConformer()
        for match in matches:
            idx2 = match[0]
            idx3 = match[1]
            bond = m.GetBondBetweenAtoms(idx2, idx3)
            jAtom = m.GetAtomWithIdx(idx2)
            kAtom = m.GetAtomWithIdx(idx3)
            for b1 in jAtom.GetBonds():
                if (b1.GetIdx() == bond.GetIdx()):
                    continue
                idx1 = b1.GetOtherAtomIdx(idx2)
                for b2 in kAtom.GetBonds():
                    if ((b2.GetIdx() == bond.GetIdx())
                            or (b2.GetIdx() == b1.GetIdx())):
                        continue
                    idx4 = b2.GetOtherAtomIdx(idx3)
                    # skip 3-membered rings
                    if (idx4 == idx1):
                        continue
                    # skip torsions that include hydrogens
                    #                     if ((m.GetAtomWithIdx(idx1).GetAtomicNum() == 1)
                    #                         or (m.GetAtomWithIdx(idx4).GetAtomicNum() == 1)):
                    #                         continue
                    if m.GetAtomWithIdx(idx4).IsInRing():
                        torsionList.append(
                            (idx4 + atom_counter, idx3 + atom_counter, idx2 + atom_counter, idx1 + atom_counter))
                        break
                    else:
                        torsionList.append(
                            (idx1 + atom_counter, idx2 + atom_counter, idx3 + atom_counter, idx4 + atom_counter))
                        break
                break

        atom_counter += m.GetNumAtoms()
    return torsionList

def mol_with_atom_index( mol ):
    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


def SetDihedral(conf, atom_idx, new_vale):
    rdMolTransforms.SetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3], new_vale)


def GetDihedral(conf, atom_idx):
    return rdMolTransforms.GetDihedralDeg(conf, atom_idx[0], atom_idx[1], atom_idx[2], atom_idx[3])


def GetTransformationMatrix(transformations):
    x, y, z, disp_x, disp_y, disp_z = transformations
    transMat = np.array([[np.cos(z) * np.cos(y), (np.cos(z) * np.sin(y) * np.sin(x)) - (np.sin(z) * np.cos(x)),
                          (np.cos(z) * np.sin(y) * np.cos(x)) + (np.sin(z) * np.sin(x)), disp_x],
                         [np.sin(z) * np.cos(y), (np.sin(z) * np.sin(y) * np.sin(x)) + (np.cos(z) * np.cos(x)),
                          (np.sin(z) * np.sin(y) * np.cos(x)) - (np.cos(z) * np.sin(x)), disp_y],
                         [-np.sin(y), np.cos(y) * np.sin(x), np.cos(y) * np.cos(x), disp_z],
                         [0, 0, 0, 1]], dtype=np.double)
    return transMat


def apply_changes(mol, values, rotable_bonds):
    opt_mol = copy.deepcopy(mol)
    #     opt_mol = add_rdkit_conformer(opt_mol)

    # apply rotations
    [SetDihedral(opt_mol.GetConformer(), rotable_bonds[r], values[r]) for r in range(len(rotable_bonds))]

    #     # apply transformation matrix
    #     rdMolTransforms.TransformConformer(opt_mol.GetConformer(), GetTransformationMatrix(values[:6]))

    return opt_mol
# Clockwise dihedral2 from https://stackoverflow.com/questions/20305272/dihedral-torsion-angle-from-four-points-in-cartesian-coordinates-in-python
def GetDihedralFromPointCloud(Z, atom_idx):
    p = Z[list(atom_idx)]
    b = p[:-1] - p[1:]
    b[0] *= -1 #########################
    v = np.array( [ v - (v.dot(b[1])/b[1].dot(b[1])) * b[1] for v in [b[0], b[2]] ] )
    # Normalize vectors
    v /= np.sqrt(np.einsum('...i,...i', v, v)).reshape(-1,1)
    b1 = b[1] / np.linalg.norm(b[1])
    x = np.dot(v[0], v[1])
    m = np.cross(v[0], b1)
    y = np.dot(m, v[1])
    return np.degrees(np.arctan2( y, x ))

def A_transpose_matrix(alpha):
    return np.array([[np.cos(np.radians(alpha)), np.sin(np.radians(alpha))],
                     [-np.sin(np.radians(alpha)), np.cos(np.radians(alpha))]], dtype=np.double)

def S_vec(alpha):
    return np.array([[np.cos(np.radians(alpha))],
                     [np.sin(np.radians(alpha))]], dtype=np.double)

def get_dihedral_vonMises(mol, conf, atom_idx, Z):
    Z = np.array(Z)
    v = np.zeros((2,1))
    iAtom = mol.GetAtomWithIdx(atom_idx[1])
    jAtom = mol.GetAtomWithIdx(atom_idx[2])
    k_0 = atom_idx[0]
    i = atom_idx[1]
    j = atom_idx[2]
    l_0 = atom_idx[3]
    for b1 in iAtom.GetBonds():
        k = b1.GetOtherAtomIdx(i)
        if k == j:
            continue
        for b2 in jAtom.GetBonds():
            l = b2.GetOtherAtomIdx(j)
            if l == i:
                continue
            assert k != l
            s_star = S_vec(GetDihedralFromPointCloud(Z, (k, i, j, l)))
            a_mat = A_transpose_matrix(GetDihedral(conf, (k, i, j, k_0)) + GetDihedral(conf, (l_0, i, j, l)))
            v = v + np.matmul(a_mat, s_star)
    v = v / np.linalg.norm(v)
    v = v.reshape(-1)
    return np.degrees(np.arctan2(v[1], v[0]))