
import torch
from datasets.samplers import HardSampler
from trainer.trainer import Trainer


class BindingTrainer(Trainer):
    def __init__(self, **kwargs):
        super(BindingTrainer, self).__init__(**kwargs)

    def forward_pass(self, batch):
        lig_graphs, rec_graphs, ligs_coords, recs_coords, ligs_pocket_coords, recs_pocket_coords, geometry_graphs, complex_names = tuple(
            batch)
        ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss = self.model(lig_graphs, rec_graphs, geometry_graphs,
                                                                                         complex_names=complex_names,
                                                                                         epoch=self.epoch)
        loss, loss_components = self.loss_func(ligs_coords, recs_coords, ligs_coords_pred, ligs_pocket_coords,
                                               recs_pocket_coords, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss,
                                               self.device)
        return loss, loss_components, ligs_coords_pred, ligs_coords

    def after_batch(self, ligs_coords_pred, ligs_coords, batch_indices):
        cutoff = 5
        centroid_distances = []
        for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
            centroid_distances.append(torch.linalg.norm(lig_coords_pred.mean(dim=0) - lig_coords.mean(dim=0)))
        centroid_distances = torch.tensor(centroid_distances)
        above_cutoff = torch.tensor(batch_indices)[torch.where(centroid_distances > cutoff)[0]]
        if isinstance(self.sampler, HardSampler):
            self.sampler.add_hard_indices(above_cutoff.tolist())

    def after_epoch(self):
        if isinstance(self.sampler, HardSampler):
            self.sampler.set_hard_indices()
