import os
import sys
import yaml

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from simple_cnn import SimpleCnn

# from typing import 

# Import training parameters.

class Trainer:
    optimizers = {
        'sgd'       : torch.optim.SGD,
        'adam'      : torch.optim.Adam,
        'rmsprop'   : torch.optim.RMSprop
    }
    
    loss_fns = {
        'mse'   : nn.MSELoss,
        'bce'   : nn.BCELoss,
        'ce'    : nn.CrossEntropyLoss
    }

    def __init__(self, config_path : str):
        with open(config_path) as file:
            self.config = yaml.safe_load(config_path)
    
    def train(self):
        optimizer = self.optimizers[self.config.optimizer.name]
        lr = self.lr
        criterion = self.loss_fns[self.config.loss_fn]

        model = SimpleCnn()

        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert images to tensor
            transforms.Normalize((0.5,), (0.5,))  # Normalize to range [-1, 1]
        ])

        train_dataset = datasets.MNIST(root='../../data', train=True,
                                        transform=transform, download=False)
        test_dataset = datasets.MNIST(root='../../data', train=False,
                                       transform=transform, download=False)

        train_loader = DataLoader(train_dataset,
                                   batch_size=self.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset,
                                  batch_size=self.batch_size, shuffle=False)

        model.train()
        for epoch in self.config.epochs:
            running_loss = 0.0
            for images, labels in train_loader:
                optimizer.zero_grad()  # Clear gradients
                outputs = model(images)  # Forward pass
                loss = criterion(outputs, labels)  # Compute loss
                loss.backward()  # Backward pass
                optimizer.step()  # Update weights

                running_loss += loss.item()

            print(f"Epoch {epoch+1}/{self.epochs},\
                   Loss: {running_loss/len(train_loader)}")




if __name__ == 'main':
    if len(sys.argv) < 2:
        raise IndexError("Input path to the config file.")

    trainer = Trainer(sys.argv[1])
    trainer.train()