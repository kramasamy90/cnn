import torch
# Close the writer
import torch.nn as nn
import torch.functional as F

from unet import Unet


class UnetPretrainer(nn.Module):
    '''
    Default parameters set for Fashion MNIST dataset.
    '''
    def __init__(self, num_classes, input_size=32, 
                 fc_width = 1024, use_global_pooling = False):
        super(UnetPretrainer, self).__init__()

        self.unet = Unet()
        self.input_size = input_size
        self.fc_width = fc_width
        self.use_global_pooling = use_global_pooling
        if (input_size % 32):
            raise ValueError("Input size should be a multiple of 32.")
        
        if use_global_pooling:
            conv_out_size = 1024
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        else:
            conv_out_size = int((self.input_size * self.input_size))

        self.fc1 = nn.Linear(conv_out_size, self.fc_width)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(self.fc_width, num_classes)



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
        if self.use_global_pooling:
            x = self.global_pool(x)
        print(x.shape)
        x = torch.flatten(x, 1)

        print(x.shape)
        print(self.fc_width)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x

