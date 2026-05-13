import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet
import torch.optim as optim

'''
Importing/downloading all dataset images/labels
Download set to False as all images have been downloaded locally
Input images: 224x224x3
'''

dataset = OxfordIIITPet(
    root="./data",
    download=False
)

class CNN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        '''
        4 blocks of convolution layers + pooling
        + ReLU after layers
        '''
        # Block 1

        self.convolution1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)

        # Block 2

        self.convolution2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Block 3

        self.convolution3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # Block 4

        self.convolution4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(2, stride=2)

        # ReLU - implement into the forward function

        self.relu = nn.ReLU()

