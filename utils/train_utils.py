import os
import sys
import yaml
from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

## Utility functions for monitoring.
def get_grad_norm(model):
    total_norm = 0.0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm

## Initialization
def he_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode='fan_in',
                                 nonlinearity='relu')
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

def xavier_init(module):
    if isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

class OptimizerFactory:
    optimizers = {
        'sgd'       : torch.optim.SGD,
        'adam'      : torch.optim.Adam,
        'rmsprop'   : torch.optim.RMSprop
    }

    def __init__(self, optimizer_config: dict):
        self.optimizer_name = optimizer_config['name']
        self.optimizers_params = optimizer_config['params']
    
    def get_optimizer(self, model_weights):
        return self.optimizers[self.optimizer_name](model_weights,
                                                    **self.optimizers_params)

class SchedulerFactory:
    schedulers = {
        'multisteplr': torch.optim.lr_scheduler.MultiStepLR
    }

    def __init__(self, scheduler_config):
        self.scheduler_name = scheduler_config['name']
        self.scheduler_params = scheduler_config['params']
    
    def get_scheduler(self, optimizer):
        return self.schedulers[self.scheduler_name](optimizer,
                                                    **self.scheduler_params)

class Trainer:
    loss_fns = {
        'mse_loss'   : nn.MSELoss,
        'bce_loss'   : nn.BCELoss,
        'ce_loss'    : nn.CrossEntropyLoss
    }

    def __init__(self, model, train_dataset, config):
        if isinstance(config, str):
            with open(config, 'r') as file:
                config = yaml.safe_load(file)
        self.config = config
        self.model = model
        self.train_dataset = train_dataset

        log_dir = config.get('log_dir')
        self.writer = SummaryWriter(log_dir=log_dir) if log_dir else None

        if self.writer:
            config_yaml = yaml.dump(self.config)
            self.writer.add_text('Config', f"\n{config_yaml}")

    def train(self,
              loss_fn = None,
              progress_bar = True,
              print_loss = False,
              return_intermediate_models = False):
        device = self.config.get('device', 'cpu')
        self.model.to(device)

        optimizer_factory = OptimizerFactory(self.config['optimizer'])
        optimizer = optimizer_factory.get_optimizer(self.model.parameters())

        if self.config.get('scheduler') is not None:
            scheduler_factory = SchedulerFactory(self.config['scheduler'])
            scheduler = scheduler_factory.get_scheduler(optimizer)

        if loss_fn is not None:
            criterion = loss_fn
        else:
            criterion = self.loss_fns[self.config['loss_fn']]().\
                                                    to(self.config['device'])

        train_loader = DataLoader(self.train_dataset,
                        batch_size=self.config['batch_size'], shuffle=True)
        
        if progress_bar:
            epochs_iterator = tqdm(range(self.config['epochs']))
        else:
            epochs_iterator = range(self.config['epochs'])
        
        if return_intermediate_models:
            intermediate_models = []
        
        output = {}
        best_model = self.model
        best_epoch = 0
        min_loss = torch.inf
        loss_history = []

        self.model.train()
        for epoch in epochs_iterator:
            running_loss = 0.0
            running_grad_norm = 0.0
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()

                grad_norm = get_grad_norm(self.model)
                running_grad_norm += grad_norm

                optimizer.step()

                running_loss += loss.item()
                
                if self.writer:
                    for name, param in self.model.named_parameters():
                        if param.grad is not None:
                            self.writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

            if self.config.get('scheduler') is not None:
                scheduler.step()

            epoch_loss = running_loss / len(train_loader)
            epoch_grad_norm = running_grad_norm / len(train_loader)

            if epoch_loss < min_loss:
                min_loss = epoch_loss
                best_epoch = epoch
                best_model = deepcopy(self.model)
            
            if print_loss:
                print(f"Epoch {epoch+1}/{self.config['epochs']}, Loss: {epoch_loss}")

            if return_intermediate_models:
                intermediate_models.append(self.model)

            loss_history.append(epoch_loss)

            if self.writer:
                self.writer.add_scalar('Loss/train', epoch_loss, epoch)
                self.writer.add_scalar('Gradient_Norm/global', epoch_grad_norm, epoch)

                for name, param in self.model.named_parameters():
                    self.writer.add_histogram(f'Weights/{name}', param, epoch)

        output = {
            'model' : best_model,
            'best_epoch' : best_epoch,
            'loss_history' : loss_history,
        }

        if return_intermediate_models:
            output['intermediate_models'] = intermediate_models

        if self.writer:
            self.writer.close()
        
        return output
