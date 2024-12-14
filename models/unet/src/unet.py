import torch
# Close the writer
import torch.nn as nn
import torch.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, act=None):
        super(ConvBlock, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.act = act
        if act is None:
            self.act = nn.ReLU()

        # Layers.
        self.conv1 = nn.Conv2d(
            in_channels= self.in_channel,
            out_channels= self.out_channel,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.conv2 = nn.Conv2d(
            in_channels= self.out_channel,
            out_channels= self.out_channel,
            kernel_size=3,
            stride=1,
            padding=1
        )   


    def forward(self, x):
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x
    

class UpConv(nn.Module):
    def __init__(self, in_channel, act=None):
        super(UpConv, self).__init__()
        self.in_channel = in_channel,
        self.act = act
        if act is None:
            self.act = nn.ReLU()

        self.upsample = nn.Upsample(scale_factor=2)
        self.conv = nn.Conv2d(
            in_channel,
            int(in_channel/2),
            kernel_size=3,
            stride=1,
            padding=1
        )
    

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    '''
    Default parameters set for Fashion MNIST dataset.
    '''
    def __init__(self):
        super(Unet, self).__init__()

        # Define layers.
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ## Encoder.
        self.block1 = ConvBlock(1, 64)
        self.block2 = ConvBlock(64, 128)
        self.block3 = ConvBlock(128, 256)
        self.block4 = ConvBlock(256, 512)
        self.block5 = ConvBlock(512, 1024)

        ## Decoder.
        self.block6 = ConvBlock(1024, 512)
        self.block7 = ConvBlock(512, 256)
        self.block8 = ConvBlock(256, 128)
        self.block9 = ConvBlock(128, 64)

        ## Upcovn layers.
        self.upconv5 = UpConv(1024)
        self.upconv6 = UpConv(512)
        self.upconv7 = UpConv(256)
        self.upconv8 = UpConv(128)

        self.final_conv = nn.Conv2d(
            in_channels=64,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1
        )
      

    def forward(self, x):
        x1 = self.block1(x)
        x = self.pool(x1)

        x2 = self.block2(x)
        x = self.pool(x2)

        x3 = self.block3(x)
        x = self.pool(x3)

        x4 = self.block4(x)
        x = self.pool(x4)

        x = self.block5(x)
        x = self.upconv5(x)
        x = torch.cat((x, x4), dim=1)

        x = self.block6(x)
        x = self.upconv6(x)
        x = torch.cat((x, x3), dim=1)

        x = self.block7(x)
        x = self.upconv7(x)
        x = torch.cat((x, x2), dim=1)

        x = self.block8(x)
        x = self.upconv8(x)
        x = torch.cat((x, x1), dim=1)

        x = self.block9(x)
        x = self.final_conv(x)
        return x

