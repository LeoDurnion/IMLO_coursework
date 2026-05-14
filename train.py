import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CNN

# Using CUDA if available

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Defining the image transformation (224x224)

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Loading training data

train_dataset = OxfordIIITPet(
    root="./datasets",
    split="trainval",
    download=False,
    transform=image_transform
)

# Dataloader

dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)



'''
import matplotlib.pyplot as plt

print("dataset size:", len(train_dataset))

img, label = train_dataset[0]
print("single image shape", img.shape)


images, labels = next(iter(dataloader))
print("Batch image shape:", images.shape)  
print("Batch labels shape:", labels.shape)
'''
