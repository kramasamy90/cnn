import torch
# Close the writer
import torch.nn as nn
import torch.functional as F

from unet import Unet


class UnetPretrainer(nn.Module):
    '''
    Default parameters set for Fashion MNIST dataset.
    '''
    def __init__(self, num_classes):
        super(UnetPretrainer, self).__init__()

        self.unet = Unet()

        self.fc1 = nn.Linear(1024, 1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)


    def forward(self, x):
        x = self.unet.block1(x)
        x = self.unet.pool(x)

        x = self.unet.block2(x)
        x = self.unet.pool(x)

        x = self.unet.block3(x)
        x = self.unet.pool(x)

        x = self.unet.block4(x)
        x = self.unet.pool(x)

        x = self.unet.block5(x)
        x = self.unet.pool(x)
        x = torch.flatten(x, 1)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

