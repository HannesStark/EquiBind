import copy
import inspect
import os
import shutil
from typing import Dict, Callable

import pyaml
import torch
import numpy as np
from torch.utils.tensorboard.summary import hparams

from datasets.samplers import HardSampler
from models import *  # do not remove
from trainer.lr_schedulers import WarmUpWrapper  # do not remove

from torch.optim.lr_scheduler import *  # For loading optimizer specified in config

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from commons.utils import flatten_dict, tensorboard_gradient_magnitude, move_to_device, list_detach, concat_if_list, log


class Trainer():
    def __init__(self, model, args, metrics: Dict[str, Callable], main_metric: str, device: torch.device,
                 tensorboard_functions: Dict[str, Callable] = None, optim=None, main_metric_goal: str = 'min',
                 loss_func=torch.nn.MSELoss(), scheduler_step_per_batch: bool = True, run_dir='', sampler=None):

        self.args = args
        self.device = device
        self.model = model.to(self.device)
        self.loss_func = loss_func
        self.tensorboard_functions = tensorboard_functions
        self.metrics = metrics
        self.sampler = sampler
        self.val_per_batch = args.val_per_batch
        self.main_metric = type(self.loss_func).__name__ if main_metric == 'loss' else main_metric
        self.main_metric_goal = main_metric_goal
        self.scheduler_step_per_batch = scheduler_step_per_batch
        self.initialize_optimizer(optim)
        self.initialize_scheduler()
        if args.checkpoint:
            checkpoint = torch.load(args.checkpoint, map_location=self.device)
            self.writer = SummaryWriter(os.path.dirname(args.checkpoint))
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optim.load_state_dict(checkpoint['optimizer_state_dict'])
            if self.lr_scheduler != None and checkpoint['scheduler_state_dict'] != None:
                self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.start_epoch = checkpoint['epoch']
            self.best_val_score = checkpoint['best_val_score']
            self.optim_steps = checkpoint['optim_steps']
        else:
            self.start_epoch = 1
            self.optim_steps = 0
            self.best_val_score = -np.inf if self.main_metric_goal == 'max' else np.inf  # running score to decide whether or not a new model should be saved
            self.writer = SummaryWriter(run_dir)
            shutil.copyfile(self.args.config, os.path.join(self.writer.log_dir, os.path.basename(self.args.config)))
        #for i, param_group in enumerate(self.optim.param_groups):
        #    param_group['lr'] = 0.0003
        self.epoch = self.start_epoch
        log(f'Log directory: {self.writer.log_dir}')
        self.hparams = copy.copy(args).__dict__
        for key, value in flatten_dict(self.hparams).items():
            log(f'{key}: {value}')

    def run_per_epoch_evaluations(self, loader):
        pass

    def train(self, train_loader: DataLoader, val_loader: DataLoader):
        epochs_no_improve = 0  # counts every epoch that the validation accuracy did not improve for early stopping
        for epoch in range(self.start_epoch, self.args.num_epochs + 1):  # loop over the dataset multiple times
            self.epoch = epoch
            self.model.train()
            self.predict(train_loader, optim=self.optim)

            self.model.eval()
            with torch.no_grad():
                metrics, _, _ = self.predict(val_loader)
                val_score = metrics[self.main_metric]

                if self.lr_scheduler != None and not self.scheduler_step_per_batch:
                    self.step_schedulers(metrics=val_score)

                if self.args.eval_per_epochs > 0 and epoch % self.args.eval_per_epochs == 0:
                    self.run_per_epoch_evaluations(val_loader)

                self.tensorboard_log(metrics, data_split='val', log_hparam=True, step=self.optim_steps)
                val_loss = metrics[type(self.loss_func).__name__]
                log('[Epoch %d] %s: %.6f val loss: %.6f' % (epoch, self.main_metric, val_score, val_loss))
                # save the model with the best main_metric depending on wether we want to maximize or minimize the main metric
                if val_score >= self.best_val_score and self.main_metric_goal == 'max' or val_score <= self.best_val_score and self.main_metric_goal == 'min':
                    epochs_no_improve = 0
                    self.best_val_score = val_score
                    self.save_checkpoint(epoch, checkpoint_name='best_checkpoint.pt')
                else:
                    epochs_no_improve += 1
                self.save_checkpoint(epoch, checkpoint_name='last_checkpoint.pt')
                log('Epochs with no improvement: [', epochs_no_improve, '] and the best  ', self.main_metric,
                    ' was in ', epoch - epochs_no_improve)
                if epochs_no_improve >= self.args.patience and epoch >= self.args.minimum_epochs:  # stopping criterion
                    log(f'Early stopping criterion based on -{self.main_metric}- that should be {self.main_metric_goal}-imized reached after {epoch} epochs. Best model checkpoint was in epoch {epoch - epochs_no_improve}.')
                    break
                if epoch in self.args.models_to_save:
                    shutil.copyfile(os.path.join(self.writer.log_dir, 'best_checkpoint.pt'),
                                    os.path.join(self.writer.log_dir, f'best_checkpoint_{epoch}epochs.pt'))
                self.after_epoch()
                #if val_loss > 10000:
                #    raise Exception

        # evaluate on best checkpoint
        checkpoint = torch.load(os.path.join(self.writer.log_dir, 'best_checkpoint.pt'), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.evaluation(val_loader, data_split='val_best_checkpoint')

    def forward_pass(self, batch):
        targets = batch[-1]  # the last entry of the batch tuple is always the targets
        predictions = self.model(*batch[0])  # foward the rest of the batch to the model
        loss, *loss_components = self.loss_func(predictions, targets)
        # if loss_func does not return any loss_components, we turn the empty list into None
        return loss, (loss_components if loss_components != [] else None), predictions, targets

    def process_batch(self, batch, optim):
        loss, loss_components, predictions, targets = self.forward_pass(batch)
        if optim != None:  # run backpropagation if an optimizer is provided
            loss.backward()
            if self.args.clip_grad != None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.clip_grad, norm_type=2)
            self.optim.step()
            self.after_optim_step()  # overwrite this function to do stuff before zeroing out grads
            self.optim.zero_grad()
            self.optim_steps += 1
        return loss, loss_components, list_detach(predictions), list_detach(targets)

    def predict(self, data_loader: DataLoader, optim: torch.optim.Optimizer = None, return_pred=False):
        total_metrics = {k: 0 for k in
                         list(self.metrics.keys()) + [type(self.loss_func).__name__, 'mean_pred', 'std_pred',
                                                      'mean_targets', 'std_targets']}
        epoch_targets = []
        epoch_predictions = []
        epoch_loss = 0
        for i, batch in enumerate(data_loader):
            *batch, batch_indices = move_to_device(list(batch), self.device)
            # loss components is either none, or a dict with the components of the loss function
            loss, loss_components, predictions, targets = self.process_batch(batch, optim)
            with torch.no_grad():
                if loss_components != None and i == 0:  # add loss_component keys to total_metrics
                    total_metrics.update({k: 0 for k in loss_components.keys()})
                if self.optim_steps % self.args.log_iterations == 0 and optim != None:
                    metrics = self.evaluate_metrics(predictions, targets)
                    metrics[type(self.loss_func).__name__] = loss.item()
                    metrics.update(loss_components)
                    self.tensorboard_log(metrics, data_split='train', step=self.optim_steps)
                    log('[Epoch %d; Iter %5d/%5d] %s: loss: %.7f' % (
                        self.epoch, i + 1, len(data_loader), 'train', loss.item()))
                if optim == None and self.val_per_batch:  # during validation or testing when we want to average metrics over all the data in that dataloader
                    metrics = self.evaluate_metrics(predictions, targets, val=True)
                    metrics[type(self.loss_func).__name__] = loss.item()
                    metrics.update(loss_components)
                    for key, value in metrics.items():
                        total_metrics[key] += value
                if optim == None and not self.val_per_batch or return_pred:
                    epoch_loss += loss.item()
                    epoch_targets.extend(targets if isinstance(targets, list) else [targets])
                    epoch_predictions.extend(predictions if isinstance(predictions, list) else [predictions])
                self.after_batch(predictions, targets, batch_indices)
        if optim == None:
            loader_len = len(data_loader) if len(data_loader) != 0 else 1
            if self.val_per_batch:
                total_metrics = {k: v / loader_len for k, v in total_metrics.items()}
            else:
                total_metrics = self.evaluate_metrics(epoch_predictions, epoch_targets, val=True)
                total_metrics[type(self.loss_func).__name__] = epoch_loss / loader_len
            if return_pred:
                return total_metrics, list_detach(epoch_predictions), list_detach(epoch_targets)
            else:
                return total_metrics, None, None

    def after_batch(self, predictions, targets, batch_indices):
        pass

    def after_epoch(self):
        pass

    def after_optim_step(self):
        if self.optim_steps % self.args.log_iterations == 0:
            tensorboard_gradient_magnitude(self.optim, self.writer, self.optim_steps)
        if self.lr_scheduler != None and (self.scheduler_step_per_batch or (isinstance(self.lr_scheduler,
                                                                                       WarmUpWrapper) and self.lr_scheduler.total_warmup_steps > self.lr_scheduler._step)):  # step per batch if that is what we want to do or if we are using a warmup schedule and are still in the warmup period
            self.step_schedulers()

    def evaluate_metrics(self, predictions, targets, batch=None, val=False) -> Dict[str, float]:
        metrics = {}
        metrics[f'mean_pred'] = torch.mean(concat_if_list(predictions)).item()
        metrics[f'std_pred'] = torch.std(concat_if_list(predictions)).item()
        metrics[f'mean_targets'] = torch.mean(concat_if_list(targets)).item()
        metrics[f'std_targets'] = torch.std(concat_if_list(targets)).item()
        for key, metric in self.metrics.items():
            if not hasattr(metric, 'val_only') or val:
                metrics[key] = metric(predictions, targets).item()
        return metrics

    def tensorboard_log(self, metrics, data_split: str, step: int, log_hparam: bool = False):
        metrics['epoch'] = self.epoch
        for i, param_group in enumerate(self.optim.param_groups):
            metrics[f'lr_param_group_{i}'] = param_group['lr']
        logs = {}
        for key, metric in metrics.items():
            metric_name = f'{key}/{data_split}'
            logs[metric_name] = metric
            self.writer.add_scalar(metric_name, metric, step)

        if log_hparam:  # write hyperparameters to tensorboard
            exp, ssi, sei = hparams(flatten_dict(self.hparams), flatten_dict(logs))
            self.writer.file_writer.add_summary(exp)
            self.writer.file_writer.add_summary(ssi)
            self.writer.file_writer.add_summary(sei)

    def evaluation(self, data_loader: DataLoader, data_split: str = '', return_pred=False):
        self.model.eval()
        metrics, predictions, targets = self.predict(data_loader, return_pred=return_pred)

        with open(os.path.join(self.writer.log_dir, 'evaluation_' + data_split + '.txt'), 'w') as file:
            log('Statistics on ', data_split)
            for key, value in metrics.items():
                file.write(f'{key}: {value}\n')
                log(f'{key}: {value}')
        return metrics, predictions, targets

    def initialize_optimizer(self, optim):
        self.optim = optim(self.model.parameters(), **self.args.optimizer_params)

    def step_schedulers(self, metrics=None):
        try:
            self.lr_scheduler.step(metrics=metrics)
        except:
            self.lr_scheduler.step()

    def initialize_scheduler(self):
        if self.args.lr_scheduler:  # Needs "from torch.optim.lr_scheduler import *" to work
            self.lr_scheduler = globals()[self.args.lr_scheduler](self.optim, **self.args.lr_scheduler_params)
        else:
            self.lr_scheduler = None

    def save_checkpoint(self, epoch: int, checkpoint_name: str):
        """
        Saves checkpoint of model in the logdir of the summarywriter in the used rundi
        """
        run_dir = self.writer.log_dir
        self.save_model_state(epoch, checkpoint_name)
        train_args = copy.copy(self.args)
        train_args.config = os.path.join(run_dir, os.path.basename(self.args.config))
        with open(os.path.join(run_dir, 'train_arguments.yaml'), 'w') as yaml_path:
            pyaml.dump(train_args.__dict__, yaml_path)

        # Get the class of the used model (works because of the "from models import *" calling the init.py in the models dir)
        model_class = globals()[type(self.model).__name__]
        source_code = inspect.getsource(model_class)  # Get the sourcecode of the class of the model.
        file_name = os.path.basename(inspect.getfile(model_class))
        with open(os.path.join(run_dir, file_name), "w") as f:
            f.write(source_code)

    def save_model_state(self, epoch: int, checkpoint_name: str):
        torch.save({
            'epoch': epoch,
            'best_val_score': self.best_val_score,
            'optim_steps': self.optim_steps,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optim.state_dict(),
            'scheduler_state_dict': None if self.lr_scheduler == None else self.lr_scheduler.state_dict()
        }, os.path.join(self.writer.log_dir, checkpoint_name))
