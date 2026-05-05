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