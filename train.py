import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import random

from model import CNN

# Defining the image transformation (224x224)
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalisation values
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
split_value = int(0.8 * len(train_dataset))

for i in range(len(values)):
    if i < split_value:
        training_indices.append(values[i])
    else:
        validation_indices.append(values[i])


# Compiling the information of the image and its value
training_data = Subset(train_dataset, training_indices)
validation_data = Subset(train_dataset, validation_indices)

# Dataloader
training_data_loader = DataLoader(training_data, batch_size=32, shuffle=True)
validation_data_loader = DataLoader(validation_data, batch_size=32, shuffle=False)

optimal_validation_accuracy = 0.0

cnn = CNN(num_classes=37)

# Computing the function for the loss function - cross entropy best for classification
cross_entropy = nn.CrossEntropyLoss()

# Optimsier
optimiser = optim.Adam(cnn.parameters(), lr=0.001, weight_decay=0.0001)

EPOCHS = 30

# Training loop
for epoch in range(EPOCHS):
    cnn.train()
    running_loss = 0.0

    # Training loop
    for images, labels in training_data_loader:
        
        # Forward propogation
        forward_prop = cnn(images)

        # Compute loss
        loss = cross_entropy(forward_prop, labels)

        # Backpropogation
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        # Update running loss
        running_loss += loss.item()


    # Validation 
    cnn.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in validation_data_loader:
            predicted = torch.argmax(cnn(images), dim=1)

            # Calculating model accuracy
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    validation_accuracy = 100 * correct / total
    average_loss = running_loss / len(training_data_loader)

    # Updating to save the best model:
    if validation_accuracy > optimal_validation_accuracy:
        optimal_validation_accuracy = validation_accuracy

        # Saving optimal model
        torch.save(cnn.state_dict(), "optimal_model.pth")
        print(f"New optimal saved, validation accuracy: {validation_accuracy:.2f}")

    # Epoch, loss, accuracy
    print(f"Epoch: {epoch} ; Loss: {average_loss:.4f} ;  Validation accuracy: {validation_accuracy:.2f}%")

print(f"Optimal training accuracy: {optimal_validation_accuracy:.2f}")
    