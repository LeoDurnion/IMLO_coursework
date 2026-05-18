import torch
import torch.nn as nn
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
from torch.utils.data import DataLoader

from model import CNN

# Loading model
cnn = CNN(num_classes=37)

# Test transform 
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Standard normalisation values
])

# Loading the test split
test_dataset = OxfordIIITPet(
    root="./datasets",
    split="test",
    download=False,
    transform=test_transform
)

# Dataloader
test_data_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
cnn.load_state_dict(torch.load("optimal_model.pth"))

# Validation
cnn.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in test_data_loader:   
        predicted = torch.argmax(cnn(images), dim=1)

        # Calculating model accuracy
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

# Printing final accuracy
accuracy = 100 * correct / total
print(f"Accuracy: {accuracy}")