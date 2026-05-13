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
        + Flattening 
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

        # Flattening
        self.flatten = nn.Flatten()

        # Creating fully connected layer: 256 channels * 14x14 flattened image = 50176
        # 50176 -> 512 as 512 = 2^9 -> faster
        self.connectedlayer1 = nn.Linear(50176, 512)
        
        # 512 -> 37 (breeds). Final connected layer
        self.connectedlayer2 = nn.Linear(512, num_classes)
    

    '''
    Forward function
    '''

    def forward(self, l):

        # Block 1
        l = self.pool1(self.relu(self.convolution1(l)))

        # Block 2
        l = self.pool2(self.relu(self.convolution2(l)))

        # Block 3
        l = self.pool3(self.relu(self.convolution3(l)))

        # Block 4
        l = self.pool4(self.relu(self.convolution4(l)))

        # Flattening
        l = self.flatten(l)

        # Final output
        l = self.connectedlayer1(l)
        l = self.relu(l)
        l = self.connectedlayer2(l)

        return l


