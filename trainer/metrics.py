from typing import Union, List

import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn

from commons.utils import concat_if_list


class PearsonR(nn.Module):
    """
    Takes a single target property of the QM9 dataset, denormalizes it and turns in into meV from eV if it  is an energy
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds, targets = concat_if_list(preds), concat_if_list(targets)  # concatenate tensors if list of tensors
        shifted_x = preds - torch.mean(preds, dim=0)
        shifted_y = targets - torch.mean(targets, dim=0)
        sigma_x = torch.sqrt(torch.sum(shifted_x ** 2, dim=0))
        sigma_y = torch.sqrt(torch.sum(shifted_y ** 2, dim=0))

        pearson = torch.sum(shifted_x * shifted_y, dim=0) / (sigma_x * sigma_y + 1e-8)
        pearson = torch.clamp(pearson, min=-1, max=1)
        pearson = pearson.mean()
        return pearson


class MAE(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, preds, targets):
        loss = F.l1_loss(preds, targets)
        return loss


class Rsquared(nn.Module):
    """
        Coefficient of determination/ R squared measure tells us the goodness of fit of our model.
        Rsquared = 1 means that the regression predictions perfectly fit the data.
        If Rsquared is less than 0 then our model is worse than the mean predictor.
        https://en.wikipedia.org/wiki/Coefficient_of_determination
    """

    def __init__(self):
        super().__init__()

    def forward(self, preds, targets):
        preds, targets = concat_if_list(preds), concat_if_list(targets)  # concatenate tensors if list of tensors
        total_SS = ((targets - targets.mean()) ** 2).sum()
        residual_SS = ((targets - preds) ** 2).sum()
        return 1 - residual_SS / total_SS


class RMSD(nn.Module):
    def __init__(self) -> None:
        super(RMSD, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()

class KabschRMSD(nn.Module):
    def __init__(self) -> None:
        super(KabschRMSD, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            lig_coords_pred_mean = lig_coords_pred.mean(dim=0, keepdim=True)  # (1,3)
            lig_coords_mean = lig_coords.mean(dim=0, keepdim=True)  # (1,3)

            A = (lig_coords_pred - lig_coords_pred_mean).transpose(0, 1) @ (lig_coords - lig_coords_mean)

            U, S, Vt = torch.linalg.svd(A)

            corr_mat = torch.diag(torch.tensor([1, 1, torch.sign(torch.det(A))], device=lig_coords_pred.device))
            rotation = (U @ corr_mat) @ Vt
            translation = lig_coords_pred_mean - torch.t(rotation @ lig_coords_mean.t())  # (1,3)

            lig_coords = (rotation @ lig_coords.t()).t() + translation
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.tensor(rmsds).mean()


class RMSDmedian(nn.Module):
    def __init__(self) -> None:
        super(RMSDmedian, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        return torch.median(torch.tensor(rmsds))


class RMSDfraction(nn.Module):
    def __init__(self, distance) -> None:
        super(RMSDfraction, self).__init__()
        self.distance = distance

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        rmsds = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            rmsds.append(torch.sqrt(torch.mean(torch.sum(((lig_coords_pred - lig_coords) ** 2), dim=1))))
        count = torch.tensor(rmsds) < self.distance
        return 100 * count.sum() / len(count)


class CentroidDist(nn.Module):
    def __init__(self) -> None:
        super(CentroidDist, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
                distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0)-lig_coords.mean(dim=0)))
        return torch.tensor(distances).mean()


class CentroidDistMedian(nn.Module):
    def __init__(self) -> None:
        super(CentroidDistMedian, self).__init__()

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0)-lig_coords.mean(dim=0)))
        return torch.median(torch.tensor(distances))


class CentroidDistFraction(nn.Module):
    def __init__(self, distance) -> None:
        super(CentroidDistFraction, self).__init__()
        self.distance = distance

    def forward(self, ligs_coords_pred: List[Tensor], ligs_coords: List[Tensor]) -> Tensor:
        distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0)-lig_coords.mean(dim=0)))
        count = torch.tensor(distances) < self.distance
        return 100 * count.sum() / len(count)


class MeanPredictorLoss(nn.Module):

    def __init__(self, loss_func) -> None:
        super(MeanPredictorLoss, self).__init__()
        self.loss_func = loss_func

    def forward(self, x1: Tensor, targets: Tensor) -> Tensor:
        return self.loss_func(torch.full_like(targets, targets.mean()), targets)
