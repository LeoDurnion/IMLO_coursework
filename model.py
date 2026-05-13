import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet
import torch.optim as optim

# importing/downloading all dataset images/labels

dataset = OxfordIIITPet(
    root="./data",
    download=False
)

class CNN(nn.module):
    def __init__(self, num_classes=37):
        super().__init__()

        # Convolutional layers

        self.convolution1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()

        self.convolution2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()

        self.convolution3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()

        self.convolution4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()

        self.convolution5 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()

