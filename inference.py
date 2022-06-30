import argparse
import sys

from copy import deepcopy

import os

from dgl import load_graphs

from rdkit import Chem
from rdkit.Chem import RemoveHs
from rdkit.Geometry import Point3D
from tqdm import tqdm

from commons.geometry_utils import rigid_transform_Kabsch_3D, get_torsions, get_dihedral_vonMises, apply_changes
from commons.logger import Logger
from commons.process_mols import read_molecule, get_lig_graph_revised, \
    get_rec_graph, get_geometry_graph, get_geometry_graph_ring, \
    get_receptor_inference

from train import load_model

from datasets.pdbbind import PDBBind

from commons.utils import seed_all, read_strings_from_txt

import yaml

from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove

from torch.utils.data import DataLoader

from trainer.metrics import Rsquared, MeanPredictorLoss, MAE, PearsonR, RMSD, RMSDfraction, CentroidDist, \
    CentroidDistFraction, RMSDmedian, CentroidDistMedian

# turn on for debugging C code like Segmentation Faults
import faulthandler

faulthandler.enable()


def parse_arguments(arglist = None):
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs_clean/inference.yml')
    p.add_argument('--checkpoint', type=str, help='path to .pt file in a checkpoint directory')
    p.add_argument('--output_directory', type=str, default=None, help='path where to put the predicted results')
    p.add_argument('--run_corrections', type=bool, default=False,
                   help='whether or not to run the fast point cloud ligand fitting')
    p.add_argument('--run_dirs', type=list, default=[], help='path directory with saved runs')
    p.add_argument('--fine_tune_dirs', type=list, default=[], help='path directory with saved finetuning runs')
    p.add_argument('--inference_path', type=str, help='path to some pdb files for which you want to run inference')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum numer of epochs to run')
    p.add_argument('--dataset_params', type=dict, default={},
                   help='parameters with keywords of the dataset')
    p.add_argument('--num_train', type=int, default=-1, help='n samples of the model samples to use for train')
    p.add_argument('--num_val', type=int, default=None, help='n samples of the model samples to use for validation')
    p.add_argument('--seed', type=int, default=1, help='seed for reproducibility')
    p.add_argument('--multithreaded_seeds', type=list, default=[],
                   help='if this is non empty, multiple threads will be started, training the same model but with the different seeds')
    p.add_argument('--seed_data', type=int, default=1, help='if you want to use a different seed for the datasplit')
    p.add_argument('--loss_func', type=str, default='MSELoss', help='Class name of torch.nn like [MSELoss, L1Loss]')
    p.add_argument('--loss_params', type=dict, default={}, help='parameters with keywords of the chosen loss function')
    p.add_argument('--optimizer', type=str, default='Adam', help='Class name of torch.optim like [Adam, SGD, AdamW]')
    p.add_argument('--optimizer_params', type=dict, help='parameters with keywords of the chosen optimizer like lr')
    p.add_argument('--clip_grad', type=float, default=None, help='clip gradients if magnitude is greater')
    p.add_argument('--lr_scheduler', type=str,
                   help='Class name of torch.optim.lr_scheduler like [CosineAnnealingLR, ExponentialLR, LambdaLR]')
    p.add_argument('--lr_scheduler_params', type=dict, help='parameters with keywords of the chosen lr_scheduler')
    p.add_argument('--scheduler_step_per_batch', default=True, type=bool,
                   help='step every batch if true step every epoch otherwise')
    p.add_argument('--log_iterations', type=int, default=-1,
                   help='log every log_iterations iterations (-1 for only logging after each epoch)')
    p.add_argument('--expensive_log_iterations', type=int, default=100,
                   help='frequency with which to do expensive logging operations')
    p.add_argument('--eval_per_epochs', type=int, default=0,
                   help='frequency with which to do run the function run_eval_per_epoch that can do some expensive calculations on the val set or sth like that. If this is zero, then the function will never be called')
    p.add_argument('--metrics', default=[], help='tensorboard metrics [mae, mae_denormalized, qm9_properties ...]')
    p.add_argument('--main_metric', default='loss', help='for early stopping etc.')
    p.add_argument('--main_metric_goal', type=str, default='min', help='controls early stopping. [max, min]')
    p.add_argument('--val_per_batch', type=bool, default=True,
                   help='run evaluation every batch and then average over the eval results. When running the molhiv benchmark for example, this needs to be Fale because we need to evaluate on all val data at once since the metric is rocauc')
    p.add_argument('--tensorboard_functions', default=[], help='choices of the TENSORBOARD_FUNCTIONS in utils')
    p.add_argument('--num_epochs_local_only', type=int, default=1,
                   help='when training with OptimalTransportTrainer, this specifies for how many epochs only the local predictions will get a loss')

    p.add_argument('--collate_function', default='graph_collate', help='the collate function to use for DataLoader')
    p.add_argument('--collate_params', type=dict, default={},
                   help='parameters with keywords of the chosen collate function')
    p.add_argument('--device', type=str, default='cuda', help='What device to train on: cuda or cpu')

    p.add_argument('--models_to_save', type=list, default=[],
                   help='specify after which epochs to remember the best model')

    p.add_argument('--model_type', type=str, default='MPNN', help='Classname of one of the models in the models dir')
    p.add_argument('--model_parameters', type=dict, help='dictionary of model parameters')

    p.add_argument('--trainer', type=str, default='binding', help='')
    p.add_argument('--train_sampler', type=str, default=None, help='any of pytorchs samplers or a custom sampler')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--check_se3_invariance', type=bool, default=False, help='check it instead of generating files')
    p.add_argument('--num_confs', type=int, default=1, help='num_confs if using rdkit conformers')
    p.add_argument('--use_rdkit_coords', action="store_true",
                   help='override the rkdit usage behavior of the used model')
    p.add_argument('--no_use_rdkit_coords', action="store_false", dest = "use_rdkit_coords",
                   help='override the rkdit usage behavior of the used model')

    cmdline_parser = deepcopy(p)
    args = p.parse_args(arglist)
    clear_defaults = {key: argparse.SUPPRESS for key in args.__dict__}
    cmdline_parser.set_defaults(**clear_defaults)
    cmdline_parser._defaults = {}
    cmdline_args = cmdline_parser.parse_args(arglist)
    
    return args, cmdline_args


def inference(args, tune_args=None):
    sys.stdout = Logger(logpath=os.path.join(os.path.dirname(args.checkpoint), f'inference.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(os.path.dirname(args.checkpoint), f'inference.log'), syspart=sys.stderr)
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")

    use_rdkit_coords = args.dataset_params[
        'use_rdkit_coords'] if 'use_rdkit_coords' in args.dataset_params.keys() else False
    args.dataset_params['multiple_rdkit_conformers'] = args.num_confs > 1
    args.dataset_params['num_confs'] = args.num_confs
    data = PDBBind(device=device, complex_names_path=args.test_names, **args.dataset_params)
    print('test size: ', len(data))
    model = load_model(args, data_sample=data[0], device=device, save_trajectories=args.save_trajectories)
    print('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    batch_size = args.batch_size if args.dataset_params['use_rec_atoms'] == False else 2
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)
    loader = DataLoader(data, batch_size=batch_size, collate_fn=collate_function)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    model.load_state_dict({k: v for k, v in checkpoint['model_state_dict'].items() if 'cross_coords' not in k})
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    for conformer_id in range(args.num_confs):
        all_ligs_coords_pred = []
        all_ligs_coords = []
        all_ligs_keypts = []
        all_recs_keypts = []
        all_pocket_coords = []
        all_names = []
        data.conformer_id = conformer_id
        for i, batch in tqdm(enumerate(loader)):
            with torch.no_grad():
                lig_graphs, rec_graphs, ligs_coords, recs_coords, all_rec_coords, pockets_coords_lig, geometry_graph, names, idx = tuple(
                    batch)
                # if names[0] not in ['2fxs', '2iwx', '2vw5', '2wer', '2yge', ]: continue
                ligs_coords_pred, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss = model(lig_graphs,
                                                                                                           rec_graphs,
                                                                                                           complex_names=names,
                                                                                                           epoch=0,
                                                                                                           geometry_graph=geometry_graph.to(
                                                                                                               device) if geometry_graph != None else None)
                for lig_coords_pred, lig_coords, lig_keypts, rec_keypts, rotation, translation, rec_pocket_coords in zip(
                        ligs_coords_pred, ligs_coords, ligs_keypts, recs_keypts, rotations, translations,
                        pockets_coords_lig):
                    all_ligs_coords_pred.append(lig_coords_pred.detach().cpu())
                    all_ligs_coords.append(lig_coords.detach().cpu())
                    all_ligs_keypts.append(((rotation @ (lig_keypts).T).T + translation).detach().cpu())
                    all_recs_keypts.append(rec_keypts.detach().cpu())
                    all_pocket_coords.append(rec_pocket_coords.detach().cpu())
                if translations == []:
                    for lig_coords_pred, lig_coords in zip(ligs_coords_pred, ligs_coords):
                        all_ligs_coords_pred.append(lig_coords_pred.detach().cpu())
                        all_ligs_coords.append(lig_coords.detach().cpu())
                all_names.extend(names)

        path = os.path.join(os.path.dirname(args.checkpoint),
                            f'predictions_Tune{tune_args != None}_RDKit{use_rdkit_coords}_confID{conformer_id}.pt')
        print(f'Saving predictions to {path}')
        results = {'predictions': all_ligs_coords_pred, 'targets': all_ligs_coords, 'lig_keypts': all_ligs_keypts,
                   'rec_keypts': all_recs_keypts, 'pocket_coords': all_pocket_coords, 'names': all_names}
        torch.save(results, path)
        rmsds = []
        centroid_distsH = []
        for i, (prediction, target, lig_keypts, rec_keypts, pocket_coords, name) in tqdm(enumerate(
                zip(results['predictions'], results['targets'], results['lig_keypts'], results['rec_keypts'],
                    results['pocket_coords'], results['names']))):
            coords_pred = prediction.numpy()
            coords_native = target.numpy()
            rmsd = np.sqrt(np.sum((coords_pred - coords_native) ** 2, axis=1).mean())

            centroid_distance = np.linalg.norm(coords_native.mean(axis=0) - coords_pred.mean(axis=0))
            centroid_distsH.append(centroid_distance)
            rmsds.append(rmsd)
        rmsds = np.array(rmsds)
        centroid_distsH = np.array(centroid_distsH)

        print('EquiBind-U with hydrogens inclduded in the loss')
        print('mean rmsd: ', rmsds.mean().__round__(2), ' pm ', rmsds.std().__round__(2))
        print('rmsd precentiles: ', np.percentile(rmsds, [25, 50, 75]).round(2))
        print(f'rmsds below 2: {(100 * (rmsds < 2).sum() / len(rmsds)).__round__(2)}%')
        print(f'rmsds below 5: {(100 * (rmsds < 5).sum() / len(rmsds)).__round__(2)}%')
        print('mean centroid: ', centroid_distsH.mean().__round__(2), ' pm ',
              centroid_distsH.std().__round__(2))
        print('centroid precentiles: ', np.percentile(centroid_distsH, [25, 50, 75]).round(2))
        print(f'centroid_distances below 2: {(100 * (centroid_distsH < 2).sum() / len(centroid_distsH)).__round__(2)}%')
        print(f'centroid_distances below 5: {(100 * (centroid_distsH < 5).sum() / len(centroid_distsH)).__round__(2)}%')

        if args.run_corrections:
            rdkit_graphs, _ = load_graphs(
                f'{data.processed_dir}/lig_graphs_rdkit_coords.pt')
            kabsch_rmsds = []
            rmsds = []
            centroid_distances = []
            kabsch_rmsds_optimized = []
            rmsds_optimized = []
            centroid_distances_optimized = []
            for i, (prediction, target, lig_keypts, rec_keypts, name) in tqdm(enumerate(
                    zip(results['predictions'], results['targets'], results['lig_keypts'], results['rec_keypts'],
                        results['names']))):
                lig = read_molecule(os.path.join('data/PDBBind/', name, f'{name}_ligand.sdf'), sanitize=True)
                if lig == None:  # read mol2 file if sdf file cannot be sanitized
                    lig = read_molecule(os.path.join('data/PDBBind/', name, f'{name}_ligand.mol2'), sanitize=True)

                lig_rdkit = deepcopy(lig)
                rdkit_coords = rdkit_graphs[i].ndata['new_x'].numpy()
                conf = lig_rdkit.GetConformer()
                for i in range(lig_rdkit.GetNumAtoms()):
                    x, y, z = rdkit_coords[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

                lig_rdkit = RemoveHs(lig_rdkit)

                lig = RemoveHs(lig)
                lig_equibind = deepcopy(lig)
                conf = lig_equibind.GetConformer()
                for i in range(lig_equibind.GetNumAtoms()):
                    x, y, z = prediction.numpy()[i]
                    conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

                coords_pred = lig_equibind.GetConformer().GetPositions()
                coords_native = lig.GetConformer().GetPositions()
                rmsdval = np.sqrt(np.sum((coords_pred - coords_native) ** 2, axis=1).mean())
                centroid_distance = np.linalg.norm(coords_native.mean(axis=0) - coords_pred.mean(axis=0))
                R, t = rigid_transform_Kabsch_3D(coords_pred.T, coords_native.T)
                moved_coords = (R @ (coords_pred).T).T + t.squeeze()
                kabsch_rmsd = np.sqrt(np.sum((moved_coords - coords_native) ** 2, axis=1).mean())

                Z_pt_cloud = coords_pred
                rotable_bonds = get_torsions([lig_rdkit])
                new_dihedrals = np.zeros(len(rotable_bonds))
                for idx, r in enumerate(rotable_bonds):
                    new_dihedrals[idx] = get_dihedral_vonMises(lig_rdkit, lig_rdkit.GetConformer(), r, Z_pt_cloud)
                optimized_mol = apply_changes(lig_rdkit, new_dihedrals, rotable_bonds)

                coords_pred_optimized = optimized_mol.GetConformer().GetPositions()
                R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_pred.T)
                coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()

                rmsdval_optimized = np.sqrt(np.sum((coords_pred_optimized - coords_native) ** 2, axis=1).mean())
                centroid_distance_optimized = np.linalg.norm(
                    coords_native.mean(axis=0) - coords_pred_optimized.mean(axis=0))
                R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_native.T)
                moved_coords_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()
                kabsch_rmsd_optimized = np.sqrt(np.sum((moved_coords_optimized - coords_native) ** 2, axis=1).mean())
                kabsch_rmsds.append(kabsch_rmsd)
                rmsds.append(rmsdval)
                centroid_distances.append(centroid_distance)
                kabsch_rmsds_optimized.append(kabsch_rmsd_optimized)
                rmsds_optimized.append(rmsdval_optimized)
                centroid_distances_optimized.append(centroid_distance_optimized)
            kabsch_rmsds = np.array(kabsch_rmsds)
            rmsdvals = np.array(rmsds)
            centroid_distsU = np.array(centroid_distances)
            kabsch_rmsds_optimized = np.array(kabsch_rmsds_optimized)
            rmsd_optimized = np.array(rmsds_optimized)
            centroid_dists = np.array(centroid_distances_optimized)
            print('EquiBind-U')
            print('mean rmsdval: ', rmsdvals.mean().__round__(2), ' pm ', rmsdvals.std().__round__(2))
            print('rmsd precentiles: ', np.percentile(rmsdvals, [25, 50, 75]).round(2))
            print(f'rmsdvals below 2: {(100 * (rmsdvals < 2).sum() / len(rmsdvals)).__round__(2)}%')
            print(f'rmsdvals below 5: {(100 * (rmsdvals < 5).sum() / len(rmsdvals)).__round__(2)}%')
            print('mean centroid: ', centroid_distsU.mean().__round__(2), ' pm ', centroid_distsU.std().__round__(2))
            print('centroid precentiles: ', np.percentile(centroid_distsU, [25, 50, 75]).round(2))
            print(f'centroid dist below 2: {(100 * (centroid_distsU < 2).sum() / len(centroid_distsU)).__round__(2)}%')
            print(f'centroid dist below 5: {(100 * (centroid_distsU < 5).sum() / len(centroid_distsU)).__round__(2)}%')
            print(f'mean kabsch RMSD: ', kabsch_rmsds.mean().__round__(2), ' pm ', kabsch_rmsds.std().__round__(2))
            print('kabsch RMSD percentiles: ', np.percentile(kabsch_rmsds, [25, 50, 75]).round(2))

            print('EquiBind')
            print('mean rmsdval: ', rmsd_optimized.mean().__round__(2), ' pm ', rmsd_optimized.std().__round__(2))
            print('rmsd precentiles: ', np.percentile(rmsd_optimized, [25, 50, 75]).round(2))
            print(f'rmsdvals below 2: {(100 * (rmsd_optimized < 2).sum() / len(rmsd_optimized)).__round__(2)}%')
            print(f'rmsdvals below 5: {(100 * (rmsd_optimized < 5).sum() / len(rmsd_optimized)).__round__(2)}%')
            print('mean centroid: ', centroid_dists.mean().__round__(2), ' pm ', centroid_dists.std().__round__(2))
            print('centroid precentiles: ', np.percentile(centroid_dists, [25, 50, 75]).round(2))
            print(f'centroid dist below 2: {(100 * (centroid_dists < 2).sum() / len(centroid_dists)).__round__(2)}%')
            print(f'centroid dist below 5: {(100 * (centroid_dists < 5).sum() / len(centroid_dists)).__round__(2)}%')
            print(f'mean kabsch RMSD: ', kabsch_rmsds_optimized.mean().__round__(2), ' pm ',
                  kabsch_rmsds_optimized.std().__round__(2))
            print('kabsch RMSD percentiles: ', np.percentile(kabsch_rmsds_optimized, [25, 50, 75]).round(2))


def inference_from_files(args):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = None
    all_ligs_coords_corrected = []
    all_intersection_losses = []
    all_intersection_losses_untuned = []
    all_ligs_coords_pred_untuned = []
    all_ligs_coords = []
    all_ligs_keypts = []
    all_recs_keypts = []
    all_names = []
    dp = args.dataset_params
    use_rdkit_coords = args.use_rdkit_coords if args.use_rdkit_coords != None else args.dataset_params[
        'use_rdkit_coords']
    names = os.listdir(args.inference_path) if (args.inference_path != None) else tqdm(read_strings_from_txt('data/timesplit_test'))
    for idx, name in enumerate(names):
        if not name.startswith('.'):
            print(f'\nProcessing {name}: complex {idx + 1} of {len(names)}')
            file_names = os.listdir(os.path.join(args.inference_path, name))
            rec_name = [i for i in file_names if 'rec.pdb' in i or 'protein' in i][0]
            lig_names = [i for i in file_names if 'ligand' in i]
            rec_path = os.path.join(args.inference_path, name, rec_name)
            for lig_name in lig_names:
                if not os.path.exists(os.path.join(args.inference_path, name, lig_name)):
                    raise ValueError(f'Path does not exist: {os.path.join(args.inference_path, name, lig_name)}')
                print(f'Trying to load {os.path.join(args.inference_path, name, lig_name)}')
                lig = read_molecule(os.path.join(args.inference_path, name, lig_name), sanitize=True)
                if lig != None:  # read mol2 file if sdf file cannot be sanitized
                    used_lig = os.path.join(args.inference_path, name, lig_name)
                    break
            if lig_names == []: raise ValueError(f'No ligand files found. The ligand file has to contain \'ligand\'.')
            if lig == None: raise ValueError(f'None of the ligand files could be read: {lig_names}')
            print(f'Docking the receptor {os.path.join(args.inference_path, name, rec_name)}\nTo the ligand {used_lig}')

            rec, rec_coords, c_alpha_coords, n_coords, c_coords = get_receptor_inference(rec_path)
            rec_graph = get_rec_graph(rec, rec_coords, c_alpha_coords, n_coords, c_coords,
                                  use_rec_atoms=dp['use_rec_atoms'], rec_radius=dp['rec_graph_radius'],
                                  surface_max_neighbors=dp['surface_max_neighbors'],
                                  surface_graph_cutoff=dp['surface_graph_cutoff'],
                                  surface_mesh_cutoff=dp['surface_mesh_cutoff'],
                                  c_alpha_max_neighbors=dp['c_alpha_max_neighbors'])
            lig_graph = get_lig_graph_revised(lig, name, max_neighbors=dp['lig_max_neighbors'],
                                          use_rdkit_coords=use_rdkit_coords, radius=dp['lig_graph_radius'])
            if 'geometry_regularization' in dp and dp['geometry_regularization']:
                geometry_graph = get_geometry_graph(lig)
            elif 'geometry_regularization_ring' in dp and dp['geometry_regularization_ring']:
                geometry_graph = get_geometry_graph_ring(lig)
            else:
                geometry_graph = None

            start_lig_coords = lig_graph.ndata['x']
            # Randomly rotate and translate the ligand.
            rot_T, rot_b = random_rotation_translation(translation_distance=5)
            if (use_rdkit_coords):
                lig_coords_to_move = lig_graph.ndata['new_x']
            else:
                lig_coords_to_move = lig_graph.ndata['x']
            mean_to_remove = lig_coords_to_move.mean(dim=0, keepdims=True)
            input_coords = (rot_T @ (lig_coords_to_move - mean_to_remove).T).T + rot_b
            lig_graph.ndata['new_x'] = input_coords

            if model == None:
                model = load_model(args, data_sample=(lig_graph, rec_graph), device=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()

            with torch.no_grad():
                geometry_graph = geometry_graph.to(device) if geometry_graph != None else None
                ligs_coords_pred_untuned, ligs_keypts, recs_keypts, rotations, translations, geom_reg_loss = model(
                    lig_graph.to(device), rec_graph.to(device), geometry_graph, complex_names=[name], epoch=0)

                for lig_coords_pred_untuned, lig_coords, lig_keypts, rec_keypts, rotation, translation in zip(
                        ligs_coords_pred_untuned, [start_lig_coords], ligs_keypts, recs_keypts, rotations,
                        translations, ):
                    all_intersection_losses_untuned.append(
                        compute_revised_intersection_loss(lig_coords_pred_untuned.detach().cpu(), rec_graph.ndata['x'],
                                                      alpha=0.2, beta=8, aggression=0))
                    all_ligs_coords_pred_untuned.append(lig_coords_pred_untuned.detach().cpu())
                    all_ligs_coords.append(lig_coords.detach().cpu())
                    all_ligs_keypts.append(((rotation @ (lig_keypts).T).T + translation).detach().cpu())
                    all_recs_keypts.append(rec_keypts.detach().cpu())

                if args.run_corrections:
                    prediction = ligs_coords_pred_untuned[0].detach().cpu()
                    lig_input = deepcopy(lig)
                    conf = lig_input.GetConformer()
                    for i in range(lig_input.GetNumAtoms()):
                        x, y, z = input_coords.numpy()[i]
                        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

                    lig_equibind = deepcopy(lig)
                    conf = lig_equibind.GetConformer()
                    for i in range(lig_equibind.GetNumAtoms()):
                        x, y, z = prediction.numpy()[i]
                        conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))

                    coords_pred = lig_equibind.GetConformer().GetPositions()

                    Z_pt_cloud = coords_pred
                    rotable_bonds = get_torsions([lig_input])
                    new_dihedrals = np.zeros(len(rotable_bonds))
                    for idx, r in enumerate(rotable_bonds):
                        new_dihedrals[idx] = get_dihedral_vonMises(lig_input, lig_input.GetConformer(), r, Z_pt_cloud)
                    optimized_mol = apply_changes(lig_input, new_dihedrals, rotable_bonds)

                    coords_pred_optimized = optimized_mol.GetConformer().GetPositions()
                    R, t = rigid_transform_Kabsch_3D(coords_pred_optimized.T, coords_pred.T)
                    coords_pred_optimized = (R @ (coords_pred_optimized).T).T + t.squeeze()
                    all_ligs_coords_corrected.append(coords_pred_optimized)

                    if args.output_directory:
                        if not os.path.exists(f'{args.output_directory}/{name}'):
                            os.makedirs(f'{args.output_directory}/{name}')
                        conf = optimized_mol.GetConformer()
                        for i in range(optimized_mol.GetNumAtoms()):
                            x, y, z = coords_pred_optimized[i]
                            conf.SetAtomPosition(i, Point3D(float(x), float(y), float(z)))
                            block_optimized = Chem.MolToMolBlock(optimized_mol)
                        print(f'Writing prediction to {args.output_directory}/{name}/lig_equibind_corrected.sdf')
                        with open(f'{args.output_directory}/{name}/lig_equibind_corrected.sdf', "w") as newfile:
                                newfile.write(block_optimized)
                all_names.append(name)

        path = os.path.join(os.path.dirname(args.checkpoint), f'predictions_RDKit{use_rdkit_coords}.pt')
        print(f'Saving predictions to {path}')
        results = {'corrected_predictions': all_ligs_coords_corrected, 'initial_predictions': all_ligs_coords_pred_untuned,
               'targets': all_ligs_coords, 'lig_keypts': all_ligs_keypts, 'rec_keypts': all_recs_keypts,
               'names': all_names, 'intersection_losses_untuned': all_intersection_losses_untuned,
               'intersection_losses': all_intersection_losses}
        torch.save(results, path)
    else:
        pass


if __name__ == '__main__':
    args, cmdline_args = parse_arguments()

    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                if key in cmdline_args:
                    continue
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}

    for run_dir in args.run_dirs:
        args.checkpoint = f'runs/{run_dir}/best_checkpoint.pt'
        config_dict['checkpoint'] = f'runs/{run_dir}/best_checkpoint.pt'
        # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if (key not in config_dict.keys()) and (key not in cmdline_args):
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value
        args.model_parameters['noise_initial'] = 0
        if args.inference_path == None:
            inference(args)
        else:
            inference_from_files(args)
