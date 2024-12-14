import os
import sys

from copy import deepcopy
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

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


class Trainer:
    loss_fns = {
        'mse_loss'   : nn.MSELoss,
        'bce_loss'   : nn.BCELoss,
        'ce_loss'    : nn.CrossEntropyLoss
    }


    def __init__(self, model, train_dataset, config):
        self.config = config
        self.model = model
        self.train_dataset = train_dataset

    
    def train(self,
              loss_fn = None,
              progress_bar = True, 
              print_loss = False,
              return_intermediate_models = False):

        device = self.config.get('device', 'cpu')
        self.model.to(device)

        # Get optimizer.
        optimizer_factory = OptimizerFactory(self.config['optimizer'])
        optimizer = optimizer_factory.get_optimizer(self.model.parameters())

        # Loss function.
        if loss_fn is not None:
            criterion = loss_fn
        else:
            criterion = self.loss_fns[self.config['loss_fn']]().\
                                                    to(self.config['device'])
        # Dataset.
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
            for images, labels in train_loader:
                images = images.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            
            if running_loss < min_loss:
                min_loss = running_loss
                best_epoch = epoch
                best_model = deepcopy(self.model)

            if print_loss:
                print(f"Epoch {epoch+1}/{self.config['epochs']},\
                    Loss: {running_loss/len(train_loader)}")
            if return_intermediate_models:
                intermediate_models.append(self.model)

            loss_history.append(running_loss / len(train_loader))

        output = {
            'model' : best_model,
            'best_epoch' : best_epoch,
            'loss_history' : loss_history,
        }

        if return_intermediate_models:
            output['intermediate_models'] = intermediate_models
    
        return output

