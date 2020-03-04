from __future__ import print_function

import numpy as np
import torch

import utils
# from q0_hello_mnist import SimpleCNN
from voc_dataset import VOCDataset

import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

class CaffeNet(nn.Module):
    def __init__(self, num_classes=20, inp_size=256, c_dim=3):
        '''
        -------ConvLayer-------
        conv(kernel_size, stride, out_channels, padding)
        nn.conv2d(in_channels, out_channels, kernel_size, stride, padding)
        VALID: uses only valid input data if stride such that not all data can be used --> No padding
        SAME: keeps the output size same as the input size ------------------------------> Applies necessary padding to do that

        Conv Layer Accepts a volume of size W1×H1×D1
            Requires four hyperparameters:
                Number of filters K, their spatial extent F, the stride S, the amount of zero padding P.
            Produces a volume of size W2×H2×D2 where:
                W2=(W1−F+2P)/S+1
                H2=(H1−F+2P)/S+1 (i.e. width and height are computed equally by symmetry)
                D2=K

        -------MAXPOOL-------
        max_pool(kernel_size, stride)
        nn.MaxPool2d(kernel_size, stride)
        MaxPool Accepts a volume of size W1×H1×D1
            Requires two hyperparameters:
                their spatial extent F, the stride S,
            Produces a volume of size W2×H2×D2 where:
                W2=(W1−F)/S+1
                H2=(H1−F)/S+1
                D2=D1
                
        cite: http://cs231n.github.io/convolutional-networks/
        '''
        super().__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(c_dim, 96, 11, 4, padding=0) # conv(11,4,96,'VALID')
        in_dim2 = 
        self.conv2 = nn.Conv2d(c_dim, 96, 11, 4, padding=) # conv(5,1,256,'SAME')
        self.non_linear = lambda x: F.relu(x, inpalce=True)
        self.pool1 = nn.MaxPool2d(3,2)

