# train.py

import argparse  # For command-line arguments
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json

from models.mlp import MLP
from init_utils import apply_weight_initialization

# -------------------
# 1. Parse Arguments
# -------------------
parser = argparse.ArgumentParser()
parser.add_argument('--init', type=str, default='he', choices=['he', 'orthogonal'], help="Weight initialization method")
parser.add_argument('--epochs', type=int, default=20, help="Number of training epochs")
parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
args = parser.parse_args()

# -------------------------
# 2. Device Configuration
# -------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------
# 3. Load and Normalize Dataset
# -------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts image to PyTorch tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
])

train_dataset = datasets.FashionMNIST(root='data', train=True, transform=transform, download=True)
test_dataset = datasets.FashionMNIST(root='data', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

# ----------------------------
# 4. Create and Initialize Model
# ----------------------------
model = MLP()
apply_weight_initialization(model, method=args.init)
model = model.to(device)

# -----------------------
# 5. Loss & Optimizer
# -----------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

# --------------------------
# 6. Training Loop
# --------------------------
train_losses = []
test_accuracies = []

for epoch in range(args.epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    train_losses.append(avg_loss)

    # Evaluate accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    test_accuracies.append(accuracy)

    print(f"Epoch [{epoch+1}/{args.epochs}], Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")

# --------------------------
# 7. Save Results to File
# --------------------------
results = {
    "init_method": args.init,
    "train_losses": train_losses,
    "test_accuracies": test_accuracies
}

with open(f"results_{args.init}.json", "w") as f:
    json.dump(results, f)

print(f"Training complete. Results saved to results_{args.init}.json")
