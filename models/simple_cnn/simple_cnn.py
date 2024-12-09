import torch
# Close the writer
import torch.nn as nn
import torch.functional as F


class SimpleCnn(nn.Module):
    '''
    Default parameters set for Fashion MNIST dataset.
    '''
    def __init__(self, h=28, w=28, n_channels=1, n_classes=10):
        super(SimpleCnn, self).__init__()
        self.h = h
        self.w = w
        self.n_classes = n_classes
        self.n_channels = n_channels

        # Define layers.
        self.act = nn.ReLU6()
        self.out_act = nn.Softmax()
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.conv1 = nn.Conv2d(
            in_channels=self.n_channels,
            out_channels=10,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels = 10,
            out_channels = 10,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )      

        self.fc1 = nn.Linear(
            in_features= 10 * (self.h * self.w) // 16,
            out_features= 10
        )
    

    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.act(x)
        x = self.pool(x)

        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.act(x)
        x = self.out_act(x)
        return x
