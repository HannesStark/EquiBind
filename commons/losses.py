import itertools
import math

import dgl
import ot
import torch
from torch import Tensor, nn
from torch.distributions import MultivariateNormal
from torch.nn.modules.loss import _Loss, L1Loss, MSELoss, BCEWithLogitsLoss
import numpy as np
import torch.nn.functional as F


# Ligand residue locations: a_i in R^3. Receptor: b_j in R^3
# Ligand: G_l(x) = -sigma * ln( \sum_i  exp(- ||x - a_i||^2 / sigma)  ), same for G_r(x)
# Ligand surface: x such that G_l(x) = surface_ct
# Other properties: G_l(a_i) < 0, G_l(x) = infinity if x is far from all a_i
# Intersection of ligand and receptor: points x such that G_l(x) < surface_ct && G_r(x) < surface_ct
# Intersection loss: IL = \avg_i max(0, surface_ct - G_r(a_i)) + \avg_j max(0, surface_ct - G_l(b_j))
def G_fn(protein_coords, x, sigma):
    # protein_coords: (n,3) ,  x: (m,3), output: (m,)
    e = torch.exp(- torch.sum((protein_coords.view(1, -1, 3) - x.view(-1, 1, 3)) ** 2, dim=2) / float(sigma))  # (m, n)
    return - sigma * torch.log(1e-3 + e.sum(dim=1))


def compute_body_intersection_loss(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma, surface_ct):
    loss = torch.mean(
        torch.clamp(surface_ct - G_fn(bound_receptor_repres_nodes_loc_array, model_ligand_coors_deform, sigma),
                    min=0)) + \
           torch.mean(
               torch.clamp(surface_ct - G_fn(model_ligand_coors_deform, bound_receptor_repres_nodes_loc_array, sigma),
                           min=0))
    return loss


def compute_sq_dist_mat(X_1, X_2):
    '''Computes the l2 squared cost matrix between two point cloud inputs.
    Args:
        X_1: [n, #features] point cloud, tensor
        X_2: [m, #features] point cloud, tensor
    Output:
        [n, m] matrix of the l2 distance between point pairs
    '''
    n_1, _ = X_1.size()
    n_2, _ = X_2.size()
    X_1 = X_1.view(n_1, 1, -1)
    X_2 = X_2.view(1, n_2, -1)
    squared_dist = (X_1 - X_2) ** 2
    cost_mat = torch.sum(squared_dist, dim=2)
    return cost_mat


def compute_ot_emd(cost_mat, device):
    cost_mat_detach = cost_mat.detach().cpu().numpy()
    a = np.ones([cost_mat.shape[0]]) / cost_mat.shape[0]
    b = np.ones([cost_mat.shape[1]]) / cost_mat.shape[1]
    ot_mat = ot.emd(a=a, b=b, M=cost_mat_detach, numItermax=10000)
    ot_mat_attached = torch.tensor(ot_mat, device=device, requires_grad=False).float()
    ot_dist = torch.sum(ot_mat_attached * cost_mat)
    return ot_dist, ot_mat_attached


def compute_revised_intersection_loss(lig_coords, rec_coords, alpha = 0.2, beta=8, aggression=0):
    distances = compute_sq_dist_mat(lig_coords,rec_coords)
    if aggression > 0:
        aggression_term = torch.clamp(-torch.log(torch.sqrt(distances)/aggression+0.01), min=1)
    else:
        aggression_term = 1
    distance_losses = aggression_term * torch.exp(-alpha*distances * torch.clamp(distances*4-beta, min=1))
    return distance_losses.sum()

class BindingLoss(_Loss):
    def __init__(self, ot_loss_weight=1, intersection_loss_weight=0, intersection_sigma=0, geom_reg_loss_weight=1, loss_rescale=True,
                 intersection_surface_ct=0, key_point_alignmen_loss_weight=0,revised_intersection_loss_weight=0, centroid_loss_weight=0, kabsch_rmsd_weight=0,translated_lig_kpt_ot_loss=False, revised_intersection_alpha=0.1, revised_intersection_beta=8, aggression=0) -> None:
        super(BindingLoss, self).__init__()
        self.ot_loss_weight = ot_loss_weight
        self.intersection_loss_weight = intersection_loss_weight
        self.intersection_sigma = intersection_sigma
        self.revised_intersection_loss_weight =revised_intersection_loss_weight
        self.intersection_surface_ct = intersection_surface_ct
        self.key_point_alignmen_loss_weight = key_point_alignmen_loss_weight
        self.centroid_loss_weight = centroid_loss_weight
        self.translated_lig_kpt_ot_loss= translated_lig_kpt_ot_loss
        self.kabsch_rmsd_weight = kabsch_rmsd_weight
        self.revised_intersection_alpha = revised_intersection_alpha
        self.revised_intersection_beta = revised_intersection_beta
        self.aggression =aggression
        self.loss_rescale = loss_rescale
        self.geom_reg_loss_weight = geom_reg_loss_weight
        self.mse_loss = MSELoss()

    def forward(self, ligs_coords, recs_coords, ligs_coords_pred, ligs_pocket_coords, recs_pocket_coords, ligs_keypts,
                recs_keypts, rotations, translations, geom_reg_loss, device, **kwargs):
        # Compute MSE loss for each protein individually, then average over the minibatch.
        ligs_coords_loss = 0
        recs_coords_loss = 0
        ot_loss = 0
        intersection_loss = 0
        intersection_loss_revised = 0
        keypts_loss = 0
        centroid_loss = 0
        kabsch_rmsd_loss = 0

        for i in range(len(ligs_coords_pred)):
            ## Compute average MSE loss (which is 3 times smaller than average squared RMSD)
            ligs_coords_loss = ligs_coords_loss + self.mse_loss(ligs_coords_pred[i], ligs_coords[i])

            if self.ot_loss_weight > 0:
                # Compute the OT loss for the binding pocket:
                ligand_pocket_coors = ligs_pocket_coords[i]  ##  (N, 3), N = num pocket nodes
                receptor_pocket_coors = recs_pocket_coords[i]  ##  (N, 3), N = num pocket nodes

                ## (N, K) cost matrix
                if self.translated_lig_kpt_ot_loss:
                    cost_mat_ligand = compute_sq_dist_mat(receptor_pocket_coors, (rotations[i] @ ligs_keypts[i].t()).t() + translations[i] )
                else:
                    cost_mat_ligand = compute_sq_dist_mat(ligand_pocket_coors, ligs_keypts[i])
                cost_mat_receptor = compute_sq_dist_mat(receptor_pocket_coors, recs_keypts[i])

                ot_dist, _ = compute_ot_emd(cost_mat_ligand + cost_mat_receptor, device)
                ot_loss += ot_dist
            if self.key_point_alignmen_loss_weight > 0:
                keypts_loss += self.mse_loss((rotations[i] @ ligs_keypts[i].t()).t() + translations[i],
                                             recs_keypts[i])

            if self.intersection_loss_weight > 0:
                intersection_loss = intersection_loss + compute_body_intersection_loss(ligs_coords_pred[i],
                                                                                       recs_coords[i],
                                                                                       self.intersection_sigma,
                                                                                       self.intersection_surface_ct)

            if self.revised_intersection_loss_weight > 0:
                intersection_loss_revised = intersection_loss_revised + compute_revised_intersection_loss(ligs_coords_pred[i],
                                                                                       recs_coords[i], alpha=self.revised_intersection_alpha, beta=self.revised_intersection_beta, aggression=self.aggression)

            if self.kabsch_rmsd_weight > 0:
                lig_coords_pred = ligs_coords_pred[i]
                lig_coords = ligs_coords[i]
                lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
                lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

                A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

                U, S, Vt = torch.linalg.svd(A)

                corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
                rotation = (U @ corr_mat) @ Vt
                translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)
                kabsch_rmsd_loss += self.mse_loss((rotation @ lig_coords.t()).t() + translation, lig_coords_pred)

            centroid_loss += self.mse_loss(ligs_coords_pred[i].mean(dim=0), ligs_coords[i].mean(dim=0))

        if self.loss_rescale:
            ligs_coords_loss = ligs_coords_loss / float(len(ligs_coords_pred))
            ot_loss = ot_loss / float(len(ligs_coords_pred))
            intersection_loss = intersection_loss / float(len(ligs_coords_pred))
            keypts_loss = keypts_loss / float(len(ligs_coords_pred))
            centroid_loss = centroid_loss / float(len(ligs_coords_pred))
            kabsch_rmsd_loss = kabsch_rmsd_loss / float(len(ligs_coords_pred))
            intersection_loss_revised = intersection_loss_revised / float(len(ligs_coords_pred))
            geom_reg_loss = geom_reg_loss / float(len(ligs_coords_pred))

        loss = ligs_coords_loss + self.ot_loss_weight * ot_loss + self.intersection_loss_weight * intersection_loss + keypts_loss * self.key_point_alignmen_loss_weight + centroid_loss * self.centroid_loss_weight + kabsch_rmsd_loss * self.kabsch_rmsd_weight + intersection_loss_revised *self.revised_intersection_loss_weight + geom_reg_loss*self.geom_reg_loss_weight
        return loss, {'ligs_coords_loss': ligs_coords_loss, 'recs_coords_loss': recs_coords_loss, 'ot_loss': ot_loss,
                      'intersection_loss': intersection_loss, 'keypts_loss': keypts_loss, 'centroid_loss:': centroid_loss, 'kabsch_rmsd_loss': kabsch_rmsd_loss, 'intersection_loss_revised': intersection_loss_revised, 'geom_reg_loss': geom_reg_loss}

class TorsionLoss(_Loss):
    def __init__(self) -> None:
        super(TorsionLoss, self).__init__()
        self.mse_loss = MSELoss()

    def forward(self, angles_pred, angles, masks, **kwargs):
        return self.mse_loss(angles_pred*masks,angles*masks)