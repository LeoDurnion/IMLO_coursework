import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import random

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

# Splitting dataset 80/20 into training/validation
dataset_size = len(train_dataset)

# Assigning each image a value and randomising for training/validation
values = list(range(dataset_size))
random.shuffle(values)

# Splitting the indices 80/20
training_indices = []
validation_indices = []
SPLIT_VALUE = 3000

for i in range(len(values)):
    if i < SPLIT_VALUE:
        training_indices.append(values[i])
    else:
        validation_indices.append(values[i])


# Compiling the information of the image and its value
training_data = Subset(train_dataset, training_indices)
validation_data = Subset(train_dataset, validation_indices)


# Dataloader
training_data_loader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_data_loader = DataLoader(validation_data, batch_size=32, shuffle=False)

cnn = CNN(num_classes=37)

# Computing the function for the loss function - cross entropy best for classification
cross_entropy = nn.CrossEntropyLoss()

# Optimsier - learning rate set to 0.001 as standard
optimiser = optim.Adam(cnn.parameters(), lr=0.001)


EPOCHS = 30

# Training loop
for epoch in range(EPOCHS):
    for images, labels in training_data_loader:
        
        # Training loop
        # Forward propogation
        forward_prop = cnn(images)

        # Compute loss
        loss = cross_entropy(forward_prop, labels)

        # Back propogation
        gradient = optimiser.zero_grad(set_to_none=True)
        loss.backward(gradient)
        optimiser.step()



'''
import matplotlib.pyplot as plt

print("dataset size:", len(train_dataset))

img, label = train_dataset[0]
print("single image shape", img.shape)


images, labels = next(iter(dataloader))
print("Batch image shape:", images.shape)  
print("Batch labels shape:", labels.shape)
'''
