import numpy as np
import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet
import torch.optim as optim

'''
Importing/downloading all dataset images/labels
Download set to False as all images have been downloaded locally
Input images: 224x224x3
'''

'''
dataset = OxfordIIITPet(
    root="./datasets",
    download=False
)
'''

class CNN(nn.Module):
    def __init__(self, num_classes=37):
        super().__init__()

        # Block 1
        self.convolution1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.batchnorm1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
  
        # Block 2
        self.convolution2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)

        # Block 3
        self.convolution3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.batchnorm3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, stride=2)

        # Block 4
        self.convolution4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.batchnorm4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
        self.pool4 = nn.MaxPool2d(2, stride=2)

        # Reducing to 1x1 
        self.pool = nn.AdaptiveAvgPool2d((1,1))

        # Flattening
        self.flatten = nn.Flatten()

        # Creating fully connected layers: 256 channels * 1x1 flattened image
        self.connectedlayer1 = nn.Linear(256, 128)

        self.dropout = nn.Dropout(0.5)

        self.connectedlayer2 = nn.Linear(128, num_classes)
    

    # Forward pass
    def forward(self, l):

        # Block 1
        l = self.pool1(self.relu(self.batchnorm1(self.convolution1(l))))

        # Block 2
        l = self.pool2(self.relu(self.batchnorm2(self.convolution2(l))))

        # Block 3
        l = self.pool3(self.relu(self.batchnorm3(self.convolution3(l))))

        # Block 4
        l = self.pool4(self.relu(self.batchnorm4(self.convolution4(l))))

        # Pooling
        l = self.pool(l)

        # Flattening
        l = self.flatten(l)

        # Final output
        l = self.connectedlayer1(l)
        l = self.relu(l)
        l = self.dropout(l)
        l = self.connectedlayer2(l)

        return l
