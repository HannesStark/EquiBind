import argparse
import concurrent.futures

import os
import sys
import traceback
from collections import defaultdict
from datetime import datetime


from commons.logger import Logger
from datasets.samplers import HardSampler
from trainer.binding_trainer import BindingTrainer


from datasets.pdbbind import PDBBind

from commons.utils import seed_all, get_random_indices, log

import yaml

from datasets.custom_collate import *  # do not remove
from models import *  # do not remove
from torch.nn import *  # do not remove
from torch.optim import *  # do not remove
from commons.losses import *  # do not remove
from torch.optim.lr_scheduler import *  # do not remove

from torch.utils.data import DataLoader, Subset

from trainer.metrics import Rsquared, MeanPredictorLoss, MAE, PearsonR, RMSD, RMSDfraction, CentroidDist, \
    CentroidDistFraction, RMSDmedian, CentroidDistMedian, KabschRMSD
from trainer.trainer import Trainer

# turn on for debugging for C code like Segmentation Faults
import faulthandler

faulthandler.enable()


def parse_arguments():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=argparse.FileType(mode='r'), default='configs_clean/RDKitCoords_flexible_self_docking.yml')
    p.add_argument('--experiment_name', type=str, help='name that will be added to the runs folder output')
    p.add_argument('--logdir', type=str, default='runs', help='tensorboard logdirectory')
    p.add_argument('--num_epochs', type=int, default=2500, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', type=int, default=1024, help='samples that will be processed in parallel')
    p.add_argument('--patience', type=int, default=20, help='stop training after no improvement in this many epochs')
    p.add_argument('--minimum_epochs', type=int, default=0, help='minimum number of epochs to run')
    p.add_argument('--dataset_params', type=dict, default={}, help='parameters with keywords of the dataset')
    p.add_argument('--dataset', type=str, default='pdbbind', help='which dataset to use')
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
    p.add_argument('--checkpoint', type=str, help='path to directory that contains a checkpoint to continue training')

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
    p.add_argument('--train_predictions_name', type=str, default=None, help='')
    p.add_argument('--val_predictions_name', type=str, default=None, help='')
    p.add_argument('--sampler_parameters', type=dict, help='dictionary of sampler parameters')

    p.add_argument('--eval_on_test', type=bool, default=True, help='runs evaluation on test set if true')
    p.add_argument('--pin_memory', type=bool, default=True, help='pin memory argument for pytorch dataloaders')
    p.add_argument('--num_workers', type=bool, default=0, help='num workers argument of dataloaders')

    return p.parse_args()


def get_trainer(args, model, data, device, metrics, run_dir, sampler=None):
    if args.trainer == None:
        trainer = Trainer
    elif args.trainer == 'binding':
        trainer = BindingTrainer

    return trainer(model=model, args=args, metrics=metrics, main_metric=args.main_metric,
                   main_metric_goal=args.main_metric_goal, optim=globals()[args.optimizer],
                   loss_func=globals()[args.loss_func](**args.loss_params), device=device, scheduler_step_per_batch=args.scheduler_step_per_batch,
                   run_dir=run_dir, sampler=sampler)


def load_model(args, data_sample, device, **kwargs):
    model = globals()[args.model_type](device=device,
                                           lig_input_edge_feats_dim=data_sample[0].edata['feat'].shape[1],
                                           rec_input_edge_feats_dim=data_sample[1].edata['feat'].shape[1],
                                           **args.model_parameters, **kwargs)
    return model


def train_wrapper(args):
    mp = args.model_parameters
    lp = args.loss_params
    if args.checkpoint:
        run_dir = os.path.dirname(args.checkpoint)
    else:
        if args.trainer == 'torsion':
            run_dir= f'{args.logdir}/{os.path.splitext(os.path.basename(args.config))[0]}_{args.experiment_name}_layers{mp["n_lays"]}_bs{args.batch_size}_dim{mp["iegmn_lay_hid_dim"]}_nAttH{mp["num_att_heads"]}_norm{mp["layer_norm"]}_normc{mp["layer_norm_coords"]}_normf{mp["final_h_layer_norm"]}_recAtoms{mp["use_rec_atoms"]}_numtrain{args.num_train}_{start_time}'
        else:
            run_dir = f'{args.logdir}/{os.path.splitext(os.path.basename(args.config))[0]}_{args.experiment_name}_layers{mp["n_lays"]}_bs{args.batch_size}_otL{lp["ot_loss_weight"]}_iL{lp["intersection_loss_weight"]}_dim{mp["iegmn_lay_hid_dim"]}_nAttH{mp["num_att_heads"]}_norm{mp["layer_norm"]}_normc{mp["layer_norm_coords"]}_normf{mp["final_h_layer_norm"]}_recAtoms{mp["use_rec_atoms"]}_numtrain{args.num_train}_{start_time}'
    if not os.path.exists(run_dir):
        os.mkdir(run_dir)

    sys.stdout = Logger(logpath=os.path.join(run_dir, f'log.log'), syspart=sys.stdout)
    sys.stderr = Logger(logpath=os.path.join(run_dir, f'log.log'), syspart=sys.stderr)
    return train(args, run_dir)



def train(args, run_dir):
    seed_all(args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() and args.device == 'cuda' else "cpu")
    metrics_dict = {'rsquared': Rsquared(),
                    'mean_rmsd': RMSD(),
                    'mean_centroid_distance': CentroidDist(),
                    'rmsd_less_than_2': RMSDfraction(2),
                    'rmsd_less_than_5': RMSDfraction(5),
                    'rmsd_less_than_10': RMSDfraction(10),
                    'rmsd_less_than_20': RMSDfraction(20),
                    'rmsd_less_than_50': RMSDfraction(50),
                    'median_rmsd': RMSDmedian(),
                    'median_centroid_distance': CentroidDistMedian(),
                    'centroid_distance_less_than_2': CentroidDistFraction(2),
                    'centroid_distance_less_than_5': CentroidDistFraction(5),
                    'centroid_distance_less_than_10': CentroidDistFraction(10),
                    'centroid_distance_less_than_20': CentroidDistFraction(20),
                    'centroid_distance_less_than_50': CentroidDistFraction(50),
                    'kabsch_rmsd': KabschRMSD(),
                    'mae': MAE(),
                    'pearsonr': PearsonR(),
                    'mean_predictor_loss': MeanPredictorLoss(globals()[args.loss_func](**args.loss_params)),
                    }

    train_data = PDBBind(device=device, complex_names_path=args.train_names,lig_predictions_name=args.train_predictions_name, is_train_data=True, **args.dataset_params)
    val_data = PDBBind(device=device, complex_names_path=args.val_names,lig_predictions_name=args.val_predictions_name, **args.dataset_params)

    if args.num_train != None:
        train_data = Subset(train_data, get_random_indices(len(train_data))[:args.num_train])
    if args.num_val != None:
        val_data = Subset(val_data, get_random_indices(len(val_data))[:args.num_val])

    log('train size: ', len(train_data))
    log('val size: ', len(val_data))

    model = load_model(args, data_sample=train_data[0], device=device)
    log('trainable params in model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    collate_function = globals()[args.collate_function] if args.collate_params == {} else globals()[
        args.collate_function](**args.collate_params)
    if args.train_sampler != None:
        sampler = globals()[args.train_sampler](data_source=train_data, batch_size=args.batch_size)
        train_loader = DataLoader(train_data, batch_sampler=sampler, collate_fn=collate_function,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
    else:
        sampler = None
        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, collate_fn=collate_function,
                                  pin_memory=args.pin_memory, num_workers=args.num_workers)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, collate_fn=collate_function,
                            pin_memory=args.pin_memory, num_workers=args.num_workers)

    metrics = {metric: metrics_dict[metric] for metric in args.metrics}
    trainer = get_trainer(args=args, model=model, data=train_data, device=device, metrics=metrics, run_dir=run_dir,
                          sampler=sampler)
    val_metrics, _, _ = trainer.train(train_loader, val_loader)
    if args.eval_on_test:
        test_data = PDBBind(device=device, complex_names_path=args.test_names, **args.dataset_params)
        test_loader = DataLoader(test_data, batch_size=args.batch_size, collate_fn=collate_function,
                                 pin_memory=args.pin_memory, num_workers=args.num_workers)
        log('test size: ', len(test_data))
        test_metrics, _, _ = trainer.evaluation(test_loader, data_split='test')
        return val_metrics, test_metrics, trainer.writer.log_dir
    return val_metrics


def get_arguments():
    args = parse_arguments()
    if args.config:
        config_dict = yaml.load(args.config, Loader=yaml.FullLoader)
        arg_dict = args.__dict__
        for key, value in config_dict.items():
            if isinstance(value, list):
                for v in value:
                    arg_dict[key].append(v)
            else:
                arg_dict[key] = value
        args.config = args.config.name
    else:
        config_dict = {}

    if args.checkpoint:  # overwrite args with args from checkpoint except for the args that were contained in the config file
        arg_dict = args.__dict__
        with open(os.path.join(os.path.dirname(args.checkpoint), 'train_arguments.yaml'), 'r') as arg_file:
            checkpoint_dict = yaml.load(arg_file, Loader=yaml.FullLoader)
        for key, value in checkpoint_dict.items():
            if key not in config_dict.keys():
                if isinstance(value, list):
                    for v in value:
                        arg_dict[key].append(v)
                else:
                    arg_dict[key] = value

    return args


def main_function():
    args = get_arguments()

    if args.multithreaded_seeds != []:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for seed in args.multithreaded_seeds:
                args_copy = get_arguments()
                args_copy.seed = seed
                futures.append(executor.submit(train_wrapper, args_copy))
            results = [f.result() for f in
                       futures]  # list of tuples of dictionaries with the validation results first and the test results second
        all_val_metrics = defaultdict(list)
        all_test_metrics = defaultdict(list)
        log_dirs = []
        for result in results:
            val_metrics, test_metrics, log_dir = result
            log_dirs.append(log_dir)
            for key in val_metrics.keys():
                all_val_metrics[key].append(val_metrics[key])
                all_test_metrics[key].append(test_metrics[key])
        files = [open(os.path.join(dir, 'multiple_seed_validation_statistics.txt'), 'w') for dir in log_dirs]
        print('Validation results:')
        for key, value in all_val_metrics.items():
            metric = np.array(value)
            for file in files:
                file.write(f'\n{key:}\n')
                file.write(f'mean: {metric.mean()}\n')
                file.write(f'stddev: {metric.std()}\n')
                file.write(f'stderr: {metric.std() / np.sqrt(len(metric))}\n')
                file.write(f'values: {value}\n')
            print(f'\n{key}:')
            print(f'mean: {metric.mean()}')
            print(f'stddev: {metric.std()}')
            print(f'stderr: {metric.std() / np.sqrt(len(metric))}')
            print(f'values: {value}')
        for file in files:
            file.close()
        files = [open(os.path.join(dir, 'multiple_seed_test_statistics.txt'), 'w') for dir in log_dirs]
        print('Test results:')
        for key, value in all_test_metrics.items():
            metric = np.array(value)
            for file in files:
                file.write(f'\n{key:}\n')
                file.write(f'mean: {metric.mean()}\n')
                file.write(f'stddev: {metric.std()}\n')
                file.write(f'stderr: {metric.std() / np.sqrt(len(metric))}\n')
                file.write(f'values: {value}\n')
            print(f'\n{key}:')
            print(f'mean: {metric.mean()}')
            print(f'stddev: {metric.std()}')
            print(f'stderr: {metric.std() / np.sqrt(len(metric))}')
            print(f'values: {value}')
        for file in files:
            file.close()
    else:
        train_wrapper(args)


if __name__ == '__main__':
    start_time = datetime.now().strftime('date%d-%m_time%H-%M-%S.%f')
    if not os.path.exists('logs'):
        os.mkdir('logs')
    with open(os.path.join('logs', f'{start_time}.log'), "w") as file:
        try:
            main_function()
        except Exception as e:
            traceback.print_exc(file=file)
            raise
